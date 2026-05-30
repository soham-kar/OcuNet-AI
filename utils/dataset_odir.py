"""
ODIR-5K Dataset Loader for Cataract Detection Baseline.
Filters the multi-label ODIR dataset for cataract vs normal cases.

Download from: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, List


def load_odir_cataract_only(csv_path: str, balance_classes: bool = True) -> pd.DataFrame:
    """
    Load only cataract cases from ODIR-5K for baseline replication.
    
    Args:
        csv_path: Path to train.csv or full_df.csv
        balance_classes: Whether to balance cataract/normal classes
    
    Returns:
        DataFrame with 'image_path' and 'label' (1=cataract, 0=normal)
    """
    df = pd.read_csv(csv_path)
    
    # ODIR columns: N, D, G, C, A, H, M, O
    # N = Normal, C = Cataract
    
    # Filter for cataract cases (C == 1)
    cataract_df = df[df['C'] == 1].copy()
    
    # Filter for normal cases (N == 1, no other disease)
    normal_df = df[df['N'] == 1].copy()
    
    print(f"Found {len(cataract_df)} cataract images")
    print(f"Found {len(normal_df)} normal images")
    
    if balance_classes:
        # Balance classes
        n_cataract = len(cataract_df)
        if len(normal_df) > n_cataract:
            normal_df = normal_df.sample(n=n_cataract, random_state=42)
        print(f"After balancing: {len(cataract_df)} cataract, {len(normal_df)} normal")
    
    # Combine and create binary labels
    combined_df = pd.concat([
        cataract_df.assign(label=1),
        normal_df.assign(label=0)
    ])
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df


class ODIRCataractDataset(Dataset):
    """
    ODIR-5K Dataset filtered for cataract detection baseline.
    
    Expected structure:
        data_root/
        ├── train.csv (or full_df.csv)
        ├── train/
        │   ├── 0_left.jpg
        │   ├── 0_right.jpg
        │   └── ...
        └── test/
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 384,
        balance_classes: bool = True
    ):
        """
        Args:
            data_root: Root directory containing train.csv and train/ folder
            split: "train", "val", or "test"
            transform: Image transforms
            image_size: Target image size
            balance_classes: Balance cataract/normal classes
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Find CSV file
        csv_path = self._find_csv()
        
        # Load cataract-filtered data
        self.df = load_odir_cataract_only(csv_path, balance_classes)
        
        # Add image paths
        self.df = self._add_image_paths()
        
        # Split data
        self.samples = self._split_data()
        
        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
    
    def _find_csv(self) -> str:
        """Find the label CSV file."""
        possible_names = ['train.csv', 'full_df.csv', 'ODIR-5K_Training_Annotations.csv']
        
        for name in possible_names:
            path = self.data_root / name
            if path.exists():
                return str(path)
        
        # Search recursively
        csv_files = list(self.data_root.rglob('*.csv'))
        if csv_files:
            return str(csv_files[0])
        
        raise FileNotFoundError(f"No CSV file found in {self.data_root}")
    
    def _add_image_paths(self) -> pd.DataFrame:
        """Add full image paths to dataframe."""
        df = self.df.copy()
        
        # Check for 'filename' or 'ID' columns
        if 'filename' in df.columns:
            filename_col = 'filename'
        elif 'Left-Fundus' in df.columns:
            # ODIR format with separate left/right columns
            # Create two rows per patient
            left_df = df[['Left-Fundus', 'label']].copy()
            left_df.columns = ['filename', 'label']
            
            right_df = df[['Right-Fundus', 'label']].copy()
            right_df.columns = ['filename', 'label']
            
            df = pd.concat([left_df, right_df]).reset_index(drop=True)
            filename_col = 'filename'
        else:
            # Try ID column
            df['filename'] = df['ID'].astype(str) + '_left.jpg'
            filename_col = 'filename'
        
        # Find image directory
        img_dirs = ['train', 'ODIR-5K_Training_Dataset', 'images']
        img_dir = None
        for d in img_dirs:
            if (self.data_root / d).exists():
                img_dir = self.data_root / d
                break
        
        if img_dir is None:
            img_dir = self.data_root
        
        # Add full paths
        df['image_path'] = df[filename_col].apply(lambda x: str(img_dir / x))
        
        # Filter out missing images
        df = df[df['image_path'].apply(lambda x: Path(x).exists())]
        
        return df
    
    def _split_data(self) -> List[dict]:
        """Split data into train/val/test."""
        np.random.seed(42)
        indices = np.random.permutation(len(self.df))
        
        n_train = int(0.8 * len(self.df))
        n_val = int(0.1 * len(self.df))
        
        if self.split == "train":
            selected_idx = indices[:n_train]
        elif self.split == "val":
            selected_idx = indices[n_train:n_train + n_val]
        else:  # test
            selected_idx = indices[n_train + n_val:]
        
        samples = []
        for i in selected_idx:
            row = self.df.iloc[i]
            samples.append({
                'image_path': row['image_path'],
                'label': row['label']
            })
        
        return samples
    
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
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Transform
        image = self.transform(image)
        
        return {
            'image': image,
            'binary_label': torch.tensor(sample['label'], dtype=torch.long),
            'severity_label': torch.tensor(sample['label'], dtype=torch.long),  # For compatibility
            'image_path': sample['image_path']
        }


def create_odir_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 384
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create ODIR dataloaders for cataract baseline training.
    """
    train_dataset = ODIRCataractDataset(data_root, split="train", image_size=image_size)
    val_dataset = ODIRCataractDataset(data_root, split="val", image_size=image_size)
    test_dataset = ODIRCataractDataset(data_root, split="test", image_size=image_size)
    
    print(f"ODIR Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
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
    print("ODIR Dataset Loader")
    print("="*50)
    print("\nDownload instructions:")
    print("1. Go to: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k")
    print("2. Download and extract to: data/raw/odir/")
    print("3. Run: python utils/dataset_odir.py")
    
    # Test if data exists
    test_path = Path("data/raw/odir")
    if test_path.exists():
        print(f"\nData found at {test_path}")
        ds = ODIRCataractDataset(str(test_path), split="train")
        print(f"Loaded {len(ds)} training samples")
    else:
        print(f"\nData not found at {test_path}")
        print("Please download the ODIR-5K dataset first.")
