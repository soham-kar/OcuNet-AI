"""
Balanced Sampler + Differential Augmentation for Multi-Label Fundus Classification.

Addresses severe class imbalance in ODIR-5K (4-8% positive rate per disease)
by:
  1. WeightedRandomSampler that oversamples images containing rare diseases
  2. Strong augmentation pipeline applied ONLY to minority-class images
  3. Focal-loss-ready label tensors

Usage (in your training script):
    from utils.balanced_sampler import BalancedODIRDataset, get_sample_weights

    train_df = pd.read_csv('data/raw/odir/train.csv')
    train_dataset = BalancedODIRDataset(
        df=train_df,
        img_dir='data/raw/odir/preprocessed_images',
        disease_cols=['Cataract', 'DR', 'Glaucoma', 'Myopia'],
        transform=None,          # uses internal transforms
        is_train=True,
        minority_aug_prob=0.7,   # 70% chance of strong aug for rare diseases
    )

    sample_weights = get_sample_weights(train_df, disease_cols)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df) * 2,   # each epoch sees 2x the data
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,          # <-- replaces shuffle=True
        num_workers=4,
        pin_memory=True,
    )
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# ── Albumentations (optional but recommended) ────────────────────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False
    print("Warning: albumentations not installed.  Install with:  pip install albumentations")


# ═══════════════════════════════════════════════════════════════════════════
# 1. SAMPLE-WEIGHT COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def get_sample_weights(
    df: pd.DataFrame,
    disease_cols: List[str],
    smoothing: str = "sqrt"
) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Strategy:
      - For each disease, compute inverse frequency (smoothed).
      - Each image gets weight = sum of inverse frequencies of its positive labels.
      - Images with NO diseases get a small baseline weight.

    Args:
        df: DataFrame with one binary column per disease (0/1 or bool).
        disease_cols: List of column names corresponding to diseases.
        smoothing: "sqrt" (recommended) or "linear".

    Returns:
        1-D numpy array of float weights, length = len(df).
    """
    n_total = len(df)
    pos_counts = df[disease_cols].sum(axis=0).values.astype(float)
    pos_counts = np.clip(pos_counts, 1.0, None)          # avoid div-by-zero

    if smoothing == "sqrt":
        class_weights = np.sqrt(n_total / pos_counts)
    elif smoothing == "linear":
        class_weights = n_total / pos_counts
    else:
        raise ValueError(f"Unknown smoothing: {smoothing}")

    # Normalise class weights so mean == 1.0 (keeps epoch length reasonable)
    class_weights = class_weights / class_weights.mean()

    # Per-sample weight
    sample_weights = np.zeros(n_total, dtype=float)
    for i, col in enumerate(disease_cols):
        sample_weights += df[col].values.astype(float) * class_weights[i]

    # Images with no disease get a small baseline weight (don't drop them entirely)
    no_disease_mask = sample_weights == 0
    sample_weights[no_disease_mask] = 0.1

    return sample_weights


# ═══════════════════════════════════════════════════════════════════════════
# 2. DIFFERENTIAL AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

def _build_base_transform(img_size: int = 224) -> Callable:
    """Minimal transforms for validation / majority-class images."""
    if _HAS_ALBU:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.ToFloat(max_value=255),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def _build_minority_transform(img_size: int = 224) -> Callable:
    """Aggressive transforms for images containing rare diseases."""
    if _HAS_ALBU:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.8
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.8
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.5
            ),
            A.ToFloat(max_value=255),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # torchvision fallback (less powerful but works without albumentations)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ═══════════════════════════════════════════════════════════════════════════
# 3. BALANCED ODIR DATASET
# ═══════════════════════════════════════════════════════════════════════════

class BalancedODIRDataset(Dataset):
    """
    ODIR-5K dataset with differential augmentation.

    - Images containing rare diseases (DR, Glaucoma) get strong augmentation
      with probability `minority_aug_prob`.
    - All other images get base (light) augmentation.
    - Supports both albumentations and torchvision backends.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        disease_cols: List[str] = None,
        img_size: int = 224,
        is_train: bool = True,
        minority_aug_prob: float = 0.7,
        rare_disease_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            df: DataFrame with columns for each disease (0/1) and an 'Image' column
                (or index) containing image filenames.
            img_dir: Directory containing the images.
            disease_cols: List of disease column names.  If None, auto-detected
                          from known ODIR names.
            img_size: Target square image size.
            is_train: If True, applies differential augmentation.
            minority_aug_prob: Probability (0-1) of applying strong augmentation
                               to an image that contains a rare disease.
            rare_disease_cols: Which diseases are considered "rare" and trigger
                               strong augmentation.  Default: DR, Glaucoma.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.minority_aug_prob = minority_aug_prob

        # Auto-detect disease columns if not provided
        if disease_cols is None:
            known = ['Cataract', 'DR', 'Glaucoma', 'Myopia',
                     'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
            disease_cols = [c for c in self.df.columns if c in known]
        self.disease_cols = disease_cols

        # Rare diseases that trigger strong augmentation
        if rare_disease_cols is None:
            rare_disease_cols = ['DR', 'Glaucoma', 'D', 'G']
        self.rare_disease_cols = [c for c in rare_disease_cols if c in self.disease_cols]

        # Build transforms
        self.base_transform = _build_base_transform(img_size)
        self.minority_transform = _build_minority_transform(img_size)

        # Determine image filename column
        if 'image_path' in self.df.columns:
            self.img_col = 'image_path'
        elif 'Image' in self.df.columns:
            self.img_col = 'Image'
        elif 'image' in self.df.columns:
            self.img_col = 'image'
        elif 'filename' in self.df.columns:
            self.img_col = 'filename'
        else:
            # Assume index contains filenames
            self.img_col = None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, idx: int) -> np.ndarray:
        """Load image as RGB numpy array (H, W, 3)."""
        if self.img_col is not None:
            fname = self.df.iloc[idx][self.img_col]
        else:
            fname = self.df.index[idx]

        # If fname is already an absolute path or starts with data/, use it directly
        path = Path(fname)
        if not path.is_absolute() and not str(path).startswith('data/'):
            path = self.img_dir / fname

        if not path.exists():
            # Try common extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt = path.with_suffix(ext)
                if alt.exists():
                    path = alt
                    break

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_labels(self, idx: int) -> torch.Tensor:
        """Return multi-label tensor (n_diseases,)."""
        row = self.df.iloc[idx]
        labels = [float(row.get(c, 0)) for c in self.disease_cols]
        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_image(idx)
        labels = self._get_labels(idx)

        if self.is_train:
            # Check if image contains any rare disease
            has_rare = any(
                float(self.df.iloc[idx].get(c, 0)) == 1.0
                for c in self.rare_disease_cols
            )

            if has_rare and random.random() < self.minority_aug_prob:
                # Strong augmentation for minority images
                if _HAS_ALBU:
                    transformed = self.minority_transform(image=img)
                    img = transformed['image']
                else:
                    img = self.minority_transform(transforms.ToPILImage()(img))
            else:
                # Base (light) augmentation
                if _HAS_ALBU:
                    transformed = self.base_transform(image=img)
                    img = transformed['image']
                else:
                    img = self.base_transform(transforms.ToPILImage()(img))
        else:
            # Validation: always base transform
            if _HAS_ALBU:
                transformed = self.base_transform(image=img)
                img = transformed['image']
            else:
                img = self.base_transform(transforms.ToPILImage()(img))

        return img, labels


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONVENIENCE FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_balanced_dataloader(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_dir: str,
    disease_cols: List[str] = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    minority_aug_prob: float = 0.7,
    epoch_multiplier: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    One-liner to create train (balanced sampler + differential aug) and
    val (standard) dataloaders.

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    # Train dataset with differential augmentation
    train_dataset = BalancedODIRDataset(
        df=train_df,
        img_dir=img_dir,
        disease_cols=disease_cols,
        img_size=img_size,
        is_train=True,
        minority_aug_prob=minority_aug_prob,
    )

    # Validation dataset (no strong augmentation)
    val_dataset = BalancedODIRDataset(
        df=val_df,
        img_dir=img_dir,
        disease_cols=disease_cols,
        img_size=img_size,
        is_train=False,
    )

    # Balanced sampler for training
    sample_weights = get_sample_weights(train_df, train_dataset.disease_cols)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df) * epoch_multiplier,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
