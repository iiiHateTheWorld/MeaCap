import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Flickr8kSplits:
    train: List[str]
    val: List[str]
    test: List[str]


def load_flickr8k_captions(captions_path: str, images_dir: str) -> pd.DataFrame:
    all_images = set(os.listdir(images_dir))
    data = pd.read_csv(captions_path, sep=",", header=None, names=["image", "caption"])
    data.image = data.image.apply(lambda x: x.split("jpg")[0] + "jpg")
    data["avail"] = data.image.apply(lambda x: x in all_images)
    data = data[data.avail == True]
    data = data.dropna()
    return data[["image", "caption"]]


def create_flickr8k_splits(
    data: pd.DataFrame,
    train_count: int = 6091,
    val_count: int = 1000,
    test_count: int = 1000,
    seed: int = 42,
) -> Flickr8kSplits:
    unique_images = sorted(data.image.unique())
    if train_count + val_count + test_count > len(unique_images):
        raise ValueError("Split sizes exceed the number of available images.")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_images), generator=rng).tolist()
    shuffled = [unique_images[idx] for idx in perm]
    train_images = shuffled[:train_count]
    val_images = shuffled[train_count : train_count + val_count]
    test_images = shuffled[train_count + val_count : train_count + val_count + test_count]
    return Flickr8kSplits(train=train_images, val=val_images, test=test_images)


def save_flickr8k_splits(splits: Flickr8kSplits, output_path: str) -> None:
    payload = {"train": splits.train, "val": splits.val, "test": splits.test}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_flickr8k_splits(splits_path: str) -> Flickr8kSplits:
    with open(splits_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Flickr8kSplits(train=payload["train"], val=payload["val"], test=payload["test"])


def build_caption_pairs(
    data: pd.DataFrame, image_list: List[str]
) -> List[Tuple[str, str]]:
    subset = data[data.image.isin(image_list)]
    return list(zip(subset.image.tolist(), subset.caption.tolist()))


def build_references(data: pd.DataFrame, image_list: List[str]) -> Dict[str, List[str]]:
    subset = data[data.image.isin(image_list)]
    references: Dict[str, List[str]] = {}
    for image_name, caption in zip(subset.image, subset.caption):
        references.setdefault(image_name, []).append(caption)
    return references


class Flickr8kCaptionDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        images_dir: str,
        tokenizer,
        clip_model,
        max_length: int = 64,
        cache_dir: str = None,
    ) -> None:
        self.pairs = pairs
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.max_length = max_length
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def _cache_path(self, image_name: str) -> str:
        safe_name = os.path.splitext(image_name)[0] + ".pt"
        return os.path.join(self.cache_dir, safe_name)

    def _load_image_embedding(self, image_name: str) -> torch.Tensor:
        if self.cache_dir is not None:
            cache_path = self._cache_path(image_name)
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        embedding = self.clip_model.compute_image_representation_from_image_instance(image).squeeze(0).cpu()
        if self.cache_dir is not None:
            torch.save(embedding, self._cache_path(image_name))
        return embedding

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        image_name, caption = self.pairs[index]
        embedding = self._load_image_embedding(image_name)
        tokens = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return image_name, embedding, tokens["input_ids"].squeeze(0)


def collate_flickr8k(batch, pad_token_id: int) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    image_names, embeddings, tokens = zip(*batch)
    embeddings = torch.stack(embeddings)
    lengths = [token.shape[0] for token in tokens]
    max_len = max(lengths)
    padded_tokens = torch.full((len(tokens), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(tokens), max_len), dtype=torch.long)
    for idx, token in enumerate(tokens):
        padded_tokens[idx, : token.shape[0]] = token
        attention_mask[idx, : token.shape[0]] = 1
    return list(image_names), embeddings, padded_tokens, attention_mask
