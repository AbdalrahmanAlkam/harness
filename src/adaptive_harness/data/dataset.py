"""Dataset splitting, partitioning, and serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from adaptive_harness.data.generator import SyntheticDataGenerator


@dataclass
class DatasetSplit:
    train_texts: List[str]
    train_labels: List[str]
    val_texts: List[str]
    val_labels: List[str]
    test_texts: List[str]
    test_labels: List[str]

    def summary(self) -> Dict[str, Dict[str, int]]:
        def count_labels(labels: List[str]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for l in labels:
                counts[l] = counts.get(l, 0) + 1
            return counts

        return {
            "train": count_labels(self.train_labels),
            "val": count_labels(self.val_labels),
            "test": count_labels(self.test_labels),
            "total_samples": {
                "train": len(self.train_texts),
                "val": len(self.val_texts),
                "test": len(self.test_texts),
            },
        }

    def save(self, directory: Path | str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        for split, texts, labels in [
            ("train", self.train_texts, self.train_labels),
            ("val", self.val_texts, self.val_labels),
            ("test", self.test_texts, self.test_labels),
        ]:
            df = pd.DataFrame({"text": texts, "label": labels})
            df.to_csv(p / f"{split}.csv", index=False)

    @classmethod
    def load(cls, directory: Path | str) -> DatasetSplit:
        p = Path(directory)
        splits = {}
        for split in ["train", "val", "test"]:
            file_path = p / f"{split}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing dataset split file: {file_path}")
            df = pd.read_csv(file_path)
            splits[f"{split}_texts"] = df["text"].astype(str).tolist()
            splits[f"{split}_labels"] = df["label"].astype(str).tolist()

        return cls(
            train_texts=splits["train_texts"],
            train_labels=splits["train_labels"],
            val_texts=splits["val_texts"],
            val_labels=splits["val_labels"],
            test_texts=splits["test_texts"],
            test_labels=splits["test_labels"],
        )


def build_synthetic_splits(
    samples_per_class: int = 800,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetSplit:
    """Builds stratified train, validation, and test splits from synthetic generation."""
    generator = SyntheticDataGenerator(seed=seed)
    data = generator.generate_all(samples_per_class=samples_per_class)

    texts = [t for t, l in data]
    labels = [l for t, l in data]

    # Stratified first split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_t, temp_t, train_l, temp_l = train_test_split(
        texts,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Stratified second split: val vs test
    val_proportion = val_ratio / val_test_ratio
    val_t, test_t, val_l, test_l = train_test_split(
        temp_t,
        temp_l,
        train_size=val_proportion,
        stratify=temp_l,
        random_state=seed,
    )

    return DatasetSplit(
        train_texts=train_t,
        train_labels=train_l,
        val_texts=val_t,
        val_labels=val_l,
        test_texts=test_t,
        test_labels=test_l,
    )
