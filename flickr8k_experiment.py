import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from dataset.flickr8k_dataset import (
    Flickr8kCaptionDataset,
    build_caption_pairs,
    build_references,
    collate_flickr8k,
    create_flickr8k_splits,
    load_flickr8k_captions,
    load_flickr8k_splits,
    save_flickr8k_splits,
)
from models.clip_utils import CLIP
from viecap.ClipCap import ClipCaptionPrefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MeaCap on Flickr8k.")
    parser.add_argument("--data_root", default="flickr8k", help="Path to Flickr8k root folder.")
    parser.add_argument("--images_dir", default="Images", help="Images directory name inside data_root.")
    parser.add_argument("--captions_file", default="captions.txt", help="Captions file name inside data_root.")
    parser.add_argument("--output_dir", default="runs/flickr8k", help="Output directory for logs and artifacts.")
    parser.add_argument("--splits_file", default="splits.json", help="Split filename stored in output_dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum caption token length.")
    parser.add_argument("--continuous_length", type=int, default=10, help="Soft prompt length.")
    parser.add_argument("--clip_project_length", type=int, default=10, help="CLIP projection length.")
    parser.add_argument("--num_layers", type=int, default=8, help="Mapping network layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Mapping network heads.")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", help="CLIP model name.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", default="runs/flickr8k/cache", help="Cache directory for CLIP embeddings.")
    parser.add_argument("--eval_max_new_tokens", type=int, default=32, help="Max new tokens for caption generation.")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam size for caption generation.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning.")
    parser.add_argument("--tune_lrs", nargs="*", type=float, default=[1e-5, 5e-5, 1e-4])
    parser.add_argument("--tune_batch_sizes", nargs="*", type=int, default=[8, 16, 32])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_root: str,
    images_dir: str,
    captions_file: str,
    output_dir: str,
    splits_file: str,
    tokenizer,
    clip_model,
    max_length: int,
    batch_size: int,
    cache_dir: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]], Dict[str, List[str]]]:
    captions_path = os.path.join(data_root, captions_file)
    images_path = os.path.join(data_root, images_dir)
    data = load_flickr8k_captions(captions_path, images_path)
    splits_path = os.path.join(output_dir, splits_file)
    if os.path.exists(splits_path):
        splits = load_flickr8k_splits(splits_path)
    else:
        splits = create_flickr8k_splits(data)
        save_flickr8k_splits(splits, splits_path)

    train_pairs = build_caption_pairs(data, splits.train)
    val_pairs = build_caption_pairs(data, splits.val)
    test_pairs = build_caption_pairs(data, splits.test)

    train_dataset = Flickr8kCaptionDataset(
        train_pairs, images_path, tokenizer, clip_model, max_length=max_length, cache_dir=cache_dir
    )
    val_dataset = Flickr8kCaptionDataset(
        val_pairs, images_path, tokenizer, clip_model, max_length=max_length, cache_dir=cache_dir
    )
    test_dataset = Flickr8kCaptionDataset(
        test_pairs, images_path, tokenizer, clip_model, max_length=max_length, cache_dir=cache_dir
    )

    collate_fn = lambda batch: collate_flickr8k(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    val_references = build_references(data, splits.val)
    test_references = build_references(data, splits.test)
    return train_loader, val_loader, test_loader, val_references, test_references


def compute_loss(
    model: ClipCaptionPrefix,
    embeddings: torch.Tensor,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_length: int,
    device: str,
) -> Tuple[torch.Tensor, float]:
    embeddings = embeddings.to(device)
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    prefix_mask = torch.ones((tokens.shape[0], prefix_length), device=device)
    full_mask = torch.cat((prefix_mask, attention_mask), dim=1)
    outputs = model(embeddings, tokens, mask=full_mask)
    logits = outputs.logits[:, :-1, :]
    labels = torch.full((tokens.shape[0], prefix_length + tokens.shape[1]), -100, device=device)
    labels[:, prefix_length:] = tokens
    labels = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        valid = labels != -100
        accuracy = (predictions[valid] == labels[valid]).float().mean().item() if valid.any() else 0.0
    return loss, accuracy


def train_one_epoch(
    model: ClipCaptionPrefix,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    prefix_length: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    for _, embeddings, tokens, attention_mask in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        loss, accuracy = compute_loss(model, embeddings, tokens, attention_mask, prefix_length, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy
    return total_loss / len(loader), total_accuracy / len(loader)


def evaluate_loss(
    model: ClipCaptionPrefix,
    loader: DataLoader,
    device: str,
    prefix_length: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for _, embeddings, tokens, attention_mask in tqdm(loader, desc="Evaluating", leave=False):
            loss, accuracy = compute_loss(model, embeddings, tokens, attention_mask, prefix_length, device)
            total_loss += loss.item()
            total_accuracy += accuracy
    return total_loss / len(loader), total_accuracy / len(loader)


def generate_caption(
    model: ClipCaptionPrefix,
    tokenizer,
    embedding: torch.Tensor,
    max_new_tokens: int,
    num_beams: int,
    device: str,
) -> str:
    model.eval()
    with torch.no_grad():
        embedding = embedding.unsqueeze(0).to(device)
        prefix_embeddings = model.mapping_network(embedding).view(1, model.continuous_length, -1)
        generated_ids = model.gpt.generate(
            inputs_embeds=prefix_embeddings,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()


def compute_metrics(
    predictions: Dict[str, str],
    references: Dict[str, List[str]],
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    refs = []
    hyps = []
    rouge_scores = []
    for image_name, prediction in predictions.items():
        ref_list = references.get(image_name, [])
        if not ref_list:
            continue
        refs.append([ref.split() for ref in ref_list])
        hyps.append(prediction.split())
        rouge = scorer.score(prediction, " ".join(ref_list))["rougeL"].fmeasure
        rouge_scores.append(rouge)
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie) if refs else 0.0
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    return {"bleu": bleu, "rougeL": rouge_l}


def evaluate_generation(
    model: ClipCaptionPrefix,
    loader: DataLoader,
    tokenizer,
    references: Dict[str, List[str]],
    max_new_tokens: int,
    num_beams: int,
    device: str,
) -> Dict[str, float]:
    predictions: Dict[str, str] = {}
    for image_names, embeddings, _, _ in tqdm(loader, desc="Generating", leave=False):
        for idx in range(embeddings.size(0)):
            if image_names[idx] in predictions:
                continue
            prediction = generate_caption(
                model,
                tokenizer,
                embeddings[idx],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                device=device,
            )
            predictions[image_names[idx]] = prediction
    return compute_metrics(predictions, references)


def plot_training_curves(history: List[Dict[str, float]], output_path: str) -> None:
    epochs = list(range(1, len(history) + 1))
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tuning_results(results: List[Dict[str, float]], output_path: str) -> None:
    labels = [f"lr={r['lr']},bs={r['batch_size']}" for r in results]
    bleu_scores = [r["val_bleu"] for r in results]
    rouge_scores = [r["val_rougeL"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(labels)), bleu_scores, label="BLEU", marker="o")
    ax.plot(range(len(labels)), rouge_scores, label="ROUGE-L", marker="o")
    ax.set_title("Hyperparameter Tuning")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    clip_model = CLIP(args.clip_model)
    clip_model = clip_model.to(args.device)

    train_loader, val_loader, test_loader, val_references, test_references = build_dataloaders(
        args.data_root,
        args.images_dir,
        args.captions_file,
        args.output_dir,
        args.splits_file,
        tokenizer,
        clip_model,
        args.max_length,
        args.batch_size,
        args.cache_dir,
    )

    model = ClipCaptionPrefix(
        continuous_length=args.continuous_length,
        clip_project_length=args.clip_project_length,
        clip_hidden_size=512,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        gpt_type="gpt2",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device, args.continuous_length)
        val_loss, val_acc = evaluate_loss(model, val_loader, args.device, args.continuous_length)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    plot_training_curves(history, os.path.join(args.output_dir, "training_curve.png"))
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    val_metrics = evaluate_generation(
        model,
        val_loader,
        tokenizer,
        val_references,
        args.eval_max_new_tokens,
        args.num_beams,
        args.device,
    )
    test_metrics = evaluate_generation(
        model,
        test_loader,
        tokenizer,
        test_references,
        args.eval_max_new_tokens,
        args.num_beams,
        args.device,
    )
    results = {
        "val_bleu": val_metrics["bleu"],
        "val_rougeL": val_metrics["rougeL"],
        "test_bleu": test_metrics["bleu"],
        "test_rougeL": test_metrics["rougeL"],
    }
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def run_tuning(args: argparse.Namespace) -> None:
    results = []
    for lr in args.tune_lrs:
        for batch_size in args.tune_batch_sizes:
            tune_args = argparse.Namespace(**vars(args))
            tune_args.lr = lr
            tune_args.batch_size = batch_size
            tune_args.output_dir = os.path.join(args.output_dir, f"lr_{lr}_bs_{batch_size}")
            metrics = run_experiment(tune_args)
            metrics.update({"lr": lr, "batch_size": batch_size})
            results.append(metrics)
    with open(os.path.join(args.output_dir, "tuning_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    plot_tuning_results(results, os.path.join(args.output_dir, "tuning_summary.png"))


def main() -> None:
    args = parse_args()
    if args.tune:
        run_tuning(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
