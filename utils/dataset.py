"""
Dataset utilities for cataract detection.

Supports:
1. Mendeley Nuclear Cataract Dataset (slit-lamp images with LOCS III grades)
2. ODIR-5K Fundus Dataset (for baseline comparison)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List
import cv2


class CataractDataset(Dataset):
    """
    Dataset for slit-lamp cataract images with LOCS III grading.
    
    Expected directory structure:
        data_root/
        ├── images/
        │   ├── NO_1.jpg
        │   ├── NC_1.jpg
        │   └── ...
        └── labels.csv  (optional, or derive from filenames)
    
    Or Mendeley structure:
        data_root/
        ├── 45_degree/
        │   ├── NO_1.jpg
        │   ├── NC1_1.jpg
        │   └── ...
        └── 135_degree/
            └── ...
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
            data_root: Root directory containing images
            split: Dataset split ("train", "val", "test")
            transform: Custom transforms (if None, uses default)
            image_size: Target image size
            label_file: Path to CSV with labels (optional)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Collect image paths and labels
        self.samples = self._collect_samples(label_file)
        
        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
        
        # LOCS III grade mapping (if derived from filenames)
        self.grade_map = {
            'NO': 0,    # Nuclear Opalescence 0 (Normal)
            'NC1': 1,   # Nuclear Color 1
            'NC2': 2,
            'NC3': 3,
            'NC4': 4,
            'NC5': 5,
            'NC6': 6
        }
    
    def _collect_samples(self, label_file: Optional[str]) -> List[Dict]:
        """Collect image paths and labels."""
        samples = []
        
        if label_file and Path(label_file).exists():
            # Load from CSV
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                samples.append({
                    'image_path': self.data_root / row['image'],
                    'binary_label': row.get('cataract', 1 if row.get('severity', 0) > 0 else 0),
                    'severity_label': row.get('severity', 0)
                })
        else:
            # Infer from directory structure and filenames
            # Look for common patterns in Mendeley dataset
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            for img_path in self.data_root.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    # Parse severity from filename
                    name = img_path.stem.upper()
                    severity = self._parse_severity_from_name(name)
                    binary = 1 if severity > 0 else 0
                    
                    samples.append({
                        'image_path': img_path,
                        'binary_label': binary,
                        'severity_label': severity
                    })
        
        # Split samples
        samples = self._split_samples(samples)
        
        return samples
    
    def _parse_severity_from_name(self, name: str) -> int:
        """Parse LOCS III severity from filename."""
        # Common patterns: NO_1, NC1_1, NC2_1, etc.
        name = name.replace('_', '').replace('-', '')
        
        for grade, value in [('NC6', 6), ('NC5', 5), ('NC4', 4), 
                             ('NC3', 3), ('NC2', 2), ('NC1', 1), ('NO', 0)]:
            if grade in name:
                return value
        
        # Default: try to extract number
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
        else:  # test
            selected_idx = indices[n_train + n_val:]
        
        return [samples[i] for i in selected_idx]
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms based on split."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
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
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        return {
            'image': image,
            'binary_label': torch.tensor(sample['binary_label'], dtype=torch.long),
            'severity_label': torch.tensor(sample['severity_label'], dtype=torch.long),
            'image_path': str(sample['image_path'])
        }


class ODIRDataset(Dataset):
    """
    ODIR-5K Fundus Dataset for baseline comparison.
    Used to replicate GLAAM baseline on fundus images.
    
    Expected structure:
        data_root/
        ├── ODIR-5K_Training_Dataset/
        │   ├── 0_left.jpg
        │   ├── 0_right.jpg
        │   └── ...
        └── ODIR-5K_Training_Annotations(Updated)_V2.xlsx
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 384
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
    
    def _load_annotations(self) -> List[Dict]:
        """Load ODIR annotations and filter for cataract cases."""
        samples = []
        
        # Look for annotation file
        anno_files = list(self.data_root.glob('*.xlsx')) + list(self.data_root.glob('*.csv'))
        
        if not anno_files:
            # If no annotation file, assume all images with 'cataract' in name are positive
            image_dir = self.data_root / 'ODIR-5K_Training_Dataset'
            if not image_dir.exists():
                image_dir = self.data_root
            
            for img_path in image_dir.glob('*.jpg'):
                samples.append({
                    'image_path': img_path,
                    'binary_label': 0  # Default to normal
                })
        else:
            # Load from annotation file
            anno_file = anno_files[0]
            if anno_file.suffix == '.xlsx':
                df = pd.read_excel(anno_file)
            else:
                df = pd.read_csv(anno_file)
            
            # ODIR has columns: ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus,
            # N, D, G, C, A, H, M, O (disease labels)
            # C = Cataract
            
            image_dir = self.data_root / 'ODIR-5K_Training_Dataset'
            if not image_dir.exists():
                image_dir = self.data_root
            
            for _, row in df.iterrows():
                patient_id = row.get('ID', row.get('id', None))
                if patient_id is None:
                    continue
                
                # Check cataract label
                cataract = row.get('C', row.get('cataract', 0))
                binary_label = 1 if cataract == 1 else 0
                
                # Left eye
                left_path = image_dir / f"{patient_id}_left.jpg"
                if left_path.exists():
                    samples.append({
                        'image_path': left_path,
                        'binary_label': binary_label
                    })
                
                # Right eye
                right_path = image_dir / f"{patient_id}_right.jpg"
                if right_path.exists():
                    samples.append({
                        'image_path': right_path,
                        'binary_label': binary_label
                    })
        
        # Split
        return self._split_samples(samples)
    
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
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'binary_label': torch.tensor(sample['binary_label'], dtype=torch.long),
            'image_path': str(sample['image_path'])
        }


def create_dataloaders(
    data_root: str,
    dataset_type: str = "slitlamp",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 384
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_root: Root directory of dataset
        dataset_type: "slitlamp" (single view), "slitlamp_dual" (45°+135° fusion), or "fundus" (ODIR)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Select dataset class based on type
    if dataset_type == "slitlamp":
        DatasetClass = CataractDataset
    elif dataset_type == "slitlamp_dual":
        # Import dual-view dataset
        try:
            from .dataset_dualview import DualViewSlitLampDataset
            DatasetClass = DualViewSlitLampDataset
            print("Using dual-view (45°+135°) dataset")
        except ImportError:
            print("Warning: DualViewSlitLampDataset not found, falling back to single view")
            DatasetClass = CataractDataset
    elif dataset_type == "fundus":
        DatasetClass = ODIRDataset
    elif dataset_type == "odir":
        # Try to use the ODIR cataract-specific loader
        try:
            from .dataset_odir import ODIRCataractDataset
            DatasetClass = ODIRCataractDataset
            print("Using ODIR-5K cataract baseline dataset")
        except ImportError:
            DatasetClass = ODIRDataset
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_dataset = DatasetClass(data_root, split="train", image_size=image_size)
    val_dataset = DatasetClass(data_root, split="val", image_size=image_size)
    test_dataset = DatasetClass(data_root, split="test", image_size=image_size)
    
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
    print("Testing dataset utilities...")
    
    # Create a dummy dataset for testing
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        for grade in ['NO', 'NC1', 'NC2', 'NC3']:
            for i in range(5):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = Path(tmpdir) / f"{grade}_{i}.jpg"
                cv2.imwrite(str(img_path), img)
        
        # Test dataset
        dataset = CataractDataset(tmpdir, split="train", image_size=224)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Binary label: {sample['binary_label']}")
            print(f"Severity label: {sample['severity_label']}")
