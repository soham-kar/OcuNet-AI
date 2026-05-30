"""
Dual-View Dataset for 45° + 135° Slit-Lamp Images.
Based on: Week 5 of 10-week roadmap - Multi-view fusion.

The Mendeley dataset has both viewing angles for each patient.
This dataset pairs them for dual-view training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List


class DualViewSlitLampDataset(Dataset):
    """
    Dataset that loads paired 45° and 135° slit-lamp views.
    
    Expected directory structure:
        data_root/
        ├── 45_degree/
        │   ├── patient001_45.jpg
        │   └── ...
        ├── 135_degree/
        │   ├── patient001_135.jpg
        │   └── ...
        └── labels.csv
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 384,
        label_file: Optional[str] = None
    ):
        """
        Args:
            data_root: Root directory containing 45_degree and 135_degree folders
            split: Dataset split ("train", "val", "test")
            transform: Image transforms
            image_size: Target image size
            label_file: Path to CSV with labels
        """
        self.data_root = Path(data_root)
        self.dir_45 = self.data_root / "45_degree"
        self.dir_135 = self.data_root / "135_degree"
        self.split = split
        self.image_size = image_size
        
        # Load samples
        self.samples = self._collect_paired_samples(label_file)
        
        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
    
    def _collect_paired_samples(self, label_file: Optional[str]) -> List[Dict]:
        """Collect paired image samples."""
        samples = []
        
        # Get all 45° images
        images_45 = set(p.stem for p in self.dir_45.glob("*.jpg"))
        images_135 = set(p.stem.replace("_135", "_45") for p in self.dir_135.glob("*.jpg"))
        
        # Find matching pairs based on patient ID
        # Assuming naming convention: patientXXX_45.jpg / patientXXX_135.jpg
        # Or: NC1_XXX.jpg in both folders (same name)
        
        if self.dir_45.exists() and self.dir_135.exists():
            for img_45 in self.dir_45.glob("*.jpg"):
                # Try to find matching 135° image
                # Strategy 1: Same filename in 135_degree folder
                img_135 = self.dir_135 / img_45.name
                
                if not img_135.exists():
                    # Strategy 2: Replace _45 with _135
                    name_135 = img_45.stem.replace("_45", "_135") + img_45.suffix
                    img_135 = self.dir_135 / name_135
                
                if img_135.exists():
                    # Parse severity from filename
                    severity = self._parse_severity_from_name(img_45.stem)
                    binary = 1 if severity > 0 else 0
                    
                    samples.append({
                        'image_45': img_45,
                        'image_135': img_135,
                        'binary_label': binary,
                        'severity_label': severity
                    })
        
        # If no pairs found, fall back to single-view
        if len(samples) == 0:
            print("Warning: No paired images found. Using single-view mode.")
            for img_path in self.dir_45.glob("*.jpg"):
                severity = self._parse_severity_from_name(img_path.stem)
                samples.append({
                    'image_45': img_path,
                    'image_135': img_path,  # Use same image
                    'binary_label': 1 if severity > 0 else 0,
                    'severity_label': severity
                })
        
        # Split
        return self._split_samples(samples)
    
    def _parse_severity_from_name(self, name: str) -> int:
        """Parse LOCS III severity from filename."""
        name = name.upper().replace('_', '').replace('-', '')
        
        for grade, value in [('NC6', 6), ('NC5', 5), ('NC4', 4), 
                             ('NC3', 3), ('NC2', 2), ('NC1', 1), ('NO', 0)]:
            if grade in name:
                return value
        return 0
    
    def _split_samples(self, samples: List[Dict]) -> List[Dict]:
        """Split samples into train/val/test."""
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        n_train = int(0.8 * len(samples))
        n_val = int(0.1 * len(samples))
        
        if self.split == "train":
            selected_idx = indices[:n_train]
        elif self.split == "val":
            selected_idx = indices[n_train:n_train + n_val]
        else:
            selected_idx = indices[n_train + n_val:]
        
        return [samples[i] for i in selected_idx]
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load both views
        image_45 = Image.open(sample['image_45']).convert('RGB')
        image_135 = Image.open(sample['image_135']).convert('RGB')
        
        # Apply same random augmentation to both
        seed = np.random.randint(2147483647)
        
        torch.manual_seed(seed)
        image_45 = self.transform(image_45)
        
        torch.manual_seed(seed)
        image_135 = self.transform(image_135)
        
        return {
            'image_45': image_45,
            'image_135': image_135,
            'binary_label': torch.tensor(sample['binary_label'], dtype=torch.long),
            'severity_label': torch.tensor(sample['severity_label'], dtype=torch.long),
            'image_path': str(sample['image_45'])
        }


def create_dualview_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 384
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for dual-view training.
    """
    train_dataset = DualViewSlitLampDataset(data_root, split="train", image_size=image_size)
    val_dataset = DualViewSlitLampDataset(data_root, split="val", image_size=image_size)
    test_dataset = DualViewSlitLampDataset(data_root, split="test", image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# For testing
if __name__ == "__main__":
    print("Testing DualViewSlitLampDataset...")
    
    # Would need actual data to test
    print("Dataset class created successfully.")
    print("To use: DualViewSlitLampDataset(data_root='data/raw/slitlamp')")
