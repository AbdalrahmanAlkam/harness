"""Data module exporting synthetic generator, dataset manager, and experience repository."""

from adaptive_harness.data.generator import SyntheticDataGenerator
from adaptive_harness.data.dataset import DatasetSplit, build_synthetic_splits
from adaptive_harness.data.storage import ExperienceRepository

__all__ = [
    "SyntheticDataGenerator",
    "DatasetSplit",
    "build_synthetic_splits",
    "ExperienceRepository",
]
