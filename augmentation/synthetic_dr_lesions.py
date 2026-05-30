"""
Synthetic DR Lesion Augmentation
Adds extracted lesion templates to healthy images during training.

Usage:
    from augmentation.synthetic_dr_lesions import SyntheticDRLesions
    
    synthetic_aug = SyntheticDRLesions(
        lesion_dir="data/synthetic/lesions/odir",
        apply_prob=0.3,
        lesions_per_image=(1, 5)
    )
    
    # In your dataset:
    image = synthetic_aug(image, is_dr_positive=False)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import random
import pickle


class SyntheticDRLesions:
    """
    Augmentation that adds synthetic DR lesions to healthy fundus images.
    
    This helps address class imbalance by:
    1. Adding microaneurysms to healthy images → creates new DR+ training samples
    2. Adding lesions to existing DR images → increases lesion diversity
    
    Args:
        lesion_dir: Directory containing lesion_*.png patches
        apply_prob: Probability of applying augmentation (default: 0.3)
        lesions_per_image: Tuple (min, max) lesions to add
        avoid_center: If True, avoid placing lesions in center (optic disc region)
    """
    
    def __init__(
        self,
        lesion_dir,
        apply_prob=0.3,
        lesions_per_image=(1, 5),
        avoid_center=True,
        blend_alpha=0.7
    ):
        self.lesion_dir = Path(lesion_dir)
        self.apply_prob = apply_prob
        self.lesions_per_image = lesions_per_image
        self.avoid_center = avoid_center
        self.blend_alpha = blend_alpha
        
        # Load lesion templates
        self.lesion_templates = []
        self._load_templates()
        
        print(f"🔬 SyntheticDRLesions initialized")
        print(f"   Lesion templates: {len(self.lesion_templates)}")
        print(f"   Apply probability: {apply_prob}")
        print(f"   Lesions per image: {lesions_per_image}")
    
    def _load_templates(self):
        """Load all lesion patches from directory"""
        if not self.lesion_dir.exists():
            print(f"⚠️  Lesion directory not found: {self.lesion_dir}")
            return
        
        for lesion_path in self.lesion_dir.glob("lesion_*.png"):
            template = cv2.imread(str(lesion_path), cv2.IMREAD_UNCHANGED)
            if template is not None:
                # Convert to RGB if needed
                if len(template.shape) == 2:
                    template = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)
                elif template.shape[2] == 4:
                    template = cv2.cvtColor(template, cv2.COLOR_BGRA2RGB)
                else:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                
                self.lesion_templates.append(template)
    
    def _get_valid_position(self, h, w, patch_size):
        """Get a random valid position for lesion placement"""
        margin = patch_size // 2
        
        if self.avoid_center:
            # Avoid center 40% of image (optic disc region)
            center_x, center_y = w // 2, h // 2
            center_margin = min(h, w) * 0.2
            
            max_attempts = 50
            for _ in range(max_attempts):
                x = random.randint(margin, w - margin - patch_size)
                y = random.randint(margin, h - margin - patch_size)
                
                # Check if outside center region
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist_from_center > center_margin:
                    return x, y
            
            # Fallback if can't find valid position
            return random.randint(margin, w - margin - patch_size), \
                   random.randint(margin, h - margin - patch_size)
        else:
            return random.randint(margin, w - margin - patch_size), \
                   random.randint(margin, h - margin - patch_size)
    
    def _augment_lesion(self, lesion):
        """Apply random augmentations to lesion template"""
        # Random rotation
        angle = random.uniform(0, 360)
        h, w = lesion.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        lesion = cv2.warpAffine(lesion, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random flip
        if random.random() > 0.5:
            lesion = cv2.flip(lesion, 1)  # Horizontal
        if random.random() > 0.5:
            lesion = cv2.flip(lesion, 0)  # Vertical
        
        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        lesion = np.clip(lesion * brightness, 0, 255).astype(np.uint8)
        
        return lesion
    
    def _blend_lesion(self, image, lesion, x, y):
        """Blend lesion onto image using alpha blending"""
        h, w = lesion.shape[:2]
        
        # Ensure we're within bounds
        if x + w > image.shape[1] or y + h > image.shape[0]:
            return image
        
        # Get the region of interest
        roi = image[y:y+h, x:x+w].copy()
        
        # Create circular mask for smooth blending
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (w//2, h//2), min(h, w)//2, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = mask[:, :, np.newaxis]
        
        # Blend
        blended = (mask * self.blend_alpha * lesion + 
                   (1 - mask * self.blend_alpha) * roi).astype(np.uint8)
        
        image[y:y+h, x:x+w] = blended
        return image
    
    def __call__(self, image, is_dr_positive=False, force_apply=False):
        """
        Apply synthetic lesions to image.
        
        Args:
            image: numpy array (H, W, 3) or torch tensor (3, H, W)
            is_dr_positive: If True, image already has DR - adds more lesions
            force_apply: If True, always apply regardless of probability
            
        Returns:
            Modified image (same format as input)
        """
        if len(self.lesion_templates) == 0:
            return image
        
        # Skip if DR positive and random chance
        if is_dr_positive and not force_apply:
            if random.random() > 0.5:  # 50% chance for DR+ images
                return image
        
        # Random probability check
        if not force_apply and random.random() > self.apply_prob:
            return image
        
        # Convert torch tensor to numpy if needed
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            # (C, H, W) -> (H, W, C)
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        h, w = image.shape[:2]
        image = image.copy()
        
        # Determine number of lesions to add
        n_lesions = random.randint(*self.lesions_per_image)
        
        for _ in range(n_lesions):
            # Select random lesion template
            lesion = random.choice(self.lesion_templates).copy()
            
            # Augment the lesion
            lesion = self._augment_lesion(lesion)
            
            # Get position
            lh, lw = lesion.shape[:2]
            x, y = self._get_valid_position(h, w, max(lh, lw))
            
            # Blend onto image
            image = self._blend_lesion(image, lesion, x, y)
        
        # Convert back to tensor if needed
        if is_tensor:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image


class SyntheticDataset:
    """
    Wrapper to add synthetic lesions to an existing dataset.
    
    Usage:
        base_dataset = MultiLabelDataset(...)
        synthetic_dataset = SyntheticDataset(
            base_dataset,
            lesion_dir="data/synthetic/lesions/odir",
            apply_prob=0.4
        )
    """
    
    def __init__(self, base_dataset, lesion_dir, apply_prob=0.4, **kwargs):
        self.base_dataset = base_dataset
        self.synthetic_aug = SyntheticDRLesions(lesion_dir, apply_prob=apply_prob, **kwargs)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Get image and label
        if isinstance(sample, tuple):
            image, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            image = sample['image']
            label = sample.get('label', sample.get('labels'))
        else:
            return sample
        
        # Check if DR positive (assuming DR is first class)
        is_dr_positive = label[0] > 0.5 if hasattr(label, '__getitem__') else False
        
        # Apply synthetic augmentation
        image = self.synthetic_aug(image, is_dr_positive=is_dr_positive)
        
        # Return in same format
        if isinstance(sample, tuple):
            return (image,) + sample[1:]
        elif isinstance(sample, dict):
            sample['image'] = image
            return sample
        
        return sample


if __name__ == "__main__":
    # Test the augmentation
    import matplotlib.pyplot as plt
    
    # Test with dummy image
    test_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    
    # Create augmentor (assuming lesions exist)
    aug = SyntheticDRLesions(
        lesion_dir="data/synthetic/lesions/odir",
        apply_prob=1.0,  # Always apply for testing
        lesions_per_image=(3, 7)
    )
    
    if len(aug.lesion_templates) > 0:
        # Apply
        augmented = aug(test_img, force_apply=True)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(test_img)
        axes[0].set_title("Original")
        axes[1].imshow(augmented)
        axes[1].set_title("With Synthetic Lesions")
        plt.savefig("synthetic_demo.png")
        print("✅ Demo saved to synthetic_demo.png")
    else:
        print("⚠️  No lesion templates found. Run extract_lesion_templates.py first.")
