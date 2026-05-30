"""
Interactive Lesion Annotation Tool for DR Synthetic Data Generation

Usage:
    python scripts/extract_lesion_templates.py \
        --image-dir data/odir/preprocessed_images \
        --label-csv data/odir/full_df.csv \
        --output-dir data/synthetic/lesions/odir \
        --disease DR \
        --patch-size 16 \
        --max-images 50

Controls:
    - Left Click: Mark lesion center
    - 's': Save all marked lesions and move to next image
    - 'n': Skip image (no lesions saved)
    - 'c': Clear last marked lesion
    - 'q': Quit annotation
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from collections import defaultdict

class InteractiveLesionExtractor:
    def __init__(self, image_dir, label_csv, output_dir, disease='DR', patch_size=16):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.disease = disease
        self.patch_size = patch_size
        self.lesion_counter = 0
        self.lesion_metadata = []
        
        # Load labels and filter positive images
        self.df = pd.read_csv(label_csv)
        self.positive_images = self._filter_positive_images()
        
        print(f"📊 Found {len(self.positive_images)} {disease}-positive images")
        print(f"📂 Output: {self.output_dir}")
        print(f"📏 Patch size: {patch_size}x{patch_size}")
    
    def _filter_positive_images(self):
        """Filter images with disease present based on CSV structure"""
        positive = []
        
        # Check column structure
        columns = self.df.columns.tolist()
        
        # ODIR format: has 'Left-Diagnostic Keywords' and 'Right-Diagnostic Keywords'
        if 'Left-Diagnostic Keywords' in columns:
            print("   Detected ODIR format")
            disease_map = {
                'DR': 'diabetic retinopathy',
                'Cataract': 'cataract', 
                'Glaucoma': 'glaucoma',
                'Myopia': 'myopia'
            }
            keyword = disease_map.get(self.disease, self.disease.lower())
            
            for _, row in self.df.iterrows():
                left_kw = str(row.get('Left-Diagnostic Keywords', '')).lower()
                right_kw = str(row.get('Right-Diagnostic Keywords', '')).lower()
                
                if keyword in left_kw:
                    left_img = row.get('Left-Fundus', '')
                    if left_img and pd.notna(left_img):
                        positive.append(left_img)
                
                if keyword in right_kw:
                    right_img = row.get('Right-Fundus', '')
                    if right_img and pd.notna(right_img):
                        positive.append(right_img)
        
        # EyePACS format: has 'level' column (0-4 severity)
        elif 'level' in columns:
            print("   Detected EyePACS format")
            # DR positive: level > 0
            positive_df = self.df[self.df['level'] > 0]
            positive = [f"{row['image']}.jpeg" for _, row in positive_df.iterrows()]
        
        # Simple binary format: has disease column directly
        elif self.disease in columns:
            print(f"   Detected binary column format")
            positive_df = self.df[self.df[self.disease] == 1]
            # Try different column names for image path
            for col in ['image_path', 'image', 'filename', 'file']:
                if col in columns:
                    positive = positive_df[col].tolist()
                    break
        
        return positive
    
    def annotate_image(self, image_path):
        """Interactive annotation with OpenCV"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"   ❌ Could not load: {image_path}")
            return True  # Continue to next
        
        # Resize for display if too large
        max_display_size = 1200
        h, w = img.shape[:2]
        scale = min(max_display_size / max(h, w), 1.0)
        
        if scale < 1.0:
            display_img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            display_img = img.copy()
        
        # Window setup
        window_name = f"Lesion Annotation: {image_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1400, int(w * scale)), min(1000, int(h * scale)))
        
        # State for mouse callback
        state = {
            'lesions': [],
            'original': img,
            'display': display_img.copy(),
            'scale': scale
        }
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Scale back to original coordinates
                orig_x = int(x / state['scale'])
                orig_y = int(y / state['scale'])
                state['lesions'].append((orig_x, orig_y))
                
                # Draw marker
                cv2.circle(state['display'], (x, y), 5, (0, 255, 0), -1)
                cv2.circle(state['display'], (x, y), int(self.patch_size * scale / 2), (0, 255, 0), 2)
                cv2.imshow(window_name, state['display'])
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Add instructions
        instructions = "Click lesions | 's'=save | 'n'=skip | 'c'=clear | 'q'=quit"
        cv2.putText(state['display'], instructions, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, state['display'])
        
        print(f"\n📷 {image_path.name} - Click on lesions, then 's' to save")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                self._save_lesions(state['original'], state['lesions'], image_path)
                cv2.destroyWindow(window_name)
                return True
            
            elif key == ord('n'):  # Skip
                print("   ⏭️  Skipped")
                cv2.destroyWindow(window_name)
                return True
            
            elif key == ord('c'):  # Clear last
                if state['lesions']:
                    state['lesions'].pop()
                    # Redraw
                    state['display'] = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
                    cv2.putText(state['display'], instructions, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    for lx, ly in state['lesions']:
                        dx, dy = int(lx * scale), int(ly * scale)
                        cv2.circle(state['display'], (dx, dy), 5, (0, 255, 0), -1)
                        cv2.circle(state['display'], (dx, dy), int(self.patch_size * scale / 2), (0, 255, 0), 2)
                    cv2.imshow(window_name, state['display'])
                    print(f"   🗑️  Cleared last ({len(state['lesions'])} remaining)")
            
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return False
    
    def _save_lesions(self, original_img, lesions, image_path):
        """Extract and save lesion patches"""
        if not lesions:
            print("   ⚠️  No lesions marked")
            return
        
        h, w = original_img.shape[:2]
        half = self.patch_size // 2
        
        for cx, cy in lesions:
            # Calculate crop boundaries
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)
            
            # Adjust if at border
            if x2 - x1 < self.patch_size:
                x1 = max(0, x2 - self.patch_size)
            if y2 - y1 < self.patch_size:
                y1 = max(0, y2 - self.patch_size)
            
            # Extract patch
            patch = original_img[y1:y2, x1:x2]
            
            # Save
            save_path = self.output_dir / f"lesion_{self.lesion_counter:04d}.png"
            cv2.imwrite(str(save_path), patch)
            
            # Metadata
            self.lesion_metadata.append({
                'id': self.lesion_counter,
                'source': str(image_path),
                'center': (cx, cy),
                'size': self.patch_size,
                'path': str(save_path)
            })
            
            self.lesion_counter += 1
        
        print(f"   ✅ Saved {len(lesions)} lesions (total: {self.lesion_counter})")
    
    def run_annotation(self, max_images=50):
        """Main annotation loop"""
        print("\n" + "="*70)
        print("🔬 Interactive Lesion Annotation Tool")
        print("="*70)
        
        processed = 0
        
        for img_name in self.positive_images[:max_images]:
            image_path = self.image_dir / img_name
            
            if not image_path.exists():
                # Try without extension variations
                for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG']:
                    alt_path = self.image_dir / (Path(img_name).stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            if not image_path.exists():
                continue
            
            should_continue = self.annotate_image(image_path)
            if not should_continue:
                break
            
            processed += 1
            print(f"📈 Progress: {processed}/{max_images}")
        
        # Save metadata
        metadata_path = self.output_dir / "lesion_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.lesion_metadata, f)
        
        print("\n" + "="*70)
        print("✅ Annotation Complete!")
        print("="*70)
        print(f"📦 Total lesions: {self.lesion_counter}")
        print(f"📝 Metadata: {metadata_path}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Interactive Lesion Annotation Tool")
    parser.add_argument("--image-dir", required=True, help="Image directory")
    parser.add_argument("--label-csv", required=True, help="Labels CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for patches")
    parser.add_argument("--disease", default="DR", choices=["DR", "Cataract", "Glaucoma", "Myopia"])
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size (default: 16)")
    parser.add_argument("--max-images", type=int, default=50, help="Max images to annotate")
    
    args = parser.parse_args()
    
    extractor = InteractiveLesionExtractor(
        image_dir=args.image_dir,
        label_csv=args.label_csv,
        output_dir=args.output_dir,
        disease=args.disease,
        patch_size=args.patch_size
    )
    
    extractor.run_annotation(max_images=args.max_images)


if __name__ == "__main__":
    main()
