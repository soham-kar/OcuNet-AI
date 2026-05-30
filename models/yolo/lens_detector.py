"""
YOLOv8 Lens Detector for Slit-lamp Images.

This module provides functionality to:
1. Detect the lens region in slit-lamp images
2. Crop the Region of Interest (ROI) for downstream classification
3. Handle cases where no lens is detected
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None


class LensDetector:
    """
    YOLOv8-based lens detector for slit-lamp images.
    
    Usage:
        detector = LensDetector()  # Uses pretrained YOLOv8n
        detector = LensDetector(model_path="path/to/trained/model.pt")  # Custom trained
        
        # Detect lens and get bbox
        bbox = detector.detect(image)
        
        # Detect and crop ROI
        roi, bbox = detector.detect_and_crop(image, target_size=(384, 384))
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to trained YOLO model. If None, uses default YOLOv8n
            conf_threshold: Confidence threshold for detection
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.is_trained = True
        else:
            # Use pretrained YOLOv8-nano (will be fine-tuned for lens detection)
            self.model = YOLO("yolov8n.pt")
            self.is_trained = False
        
        self.conf_threshold = conf_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        return_all: bool = False
    ) -> Optional[np.ndarray]:
        """
        Detect lens region in image.
        
        Args:
            image: Input image (numpy array, path, or PIL Image)
            return_all: If True, return all detections; if False, return only highest confidence
            
        Returns:
            Bounding box as [x1, y1, x2, y2] or None if no detection
            If return_all=True, returns array of shape (N, 4)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Extract boxes
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = results[0].boxes.conf.cpu().numpy()  # (N,)
        
        if return_all:
            return boxes
        else:
            # Return highest confidence detection
            best_idx = np.argmax(confs)
            return boxes[best_idx]
    
    def detect_and_crop(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        target_size: Tuple[int, int] = (384, 384),
        padding: int = 20,
        return_original_if_no_detection: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect lens and crop ROI.
        
        Args:
            image: Input image
            target_size: Size to resize the cropped ROI to (H, W)
            padding: Pixels to add around the detected box
            return_original_if_no_detection: If True, returns resized original when no detection
            
        Returns:
            roi: Cropped and resized ROI as numpy array (H, W, 3)
            bbox: Bounding box used for cropping, or None if using original
        """
        # Convert to numpy if needed
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # Detect lens
        bbox = self.detect(image)
        
        if bbox is not None:
            # Add padding
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop
            roi = image[y1:y2, x1:x2]
            bbox = np.array([x1, y1, x2, y2])
        else:
            if return_original_if_no_detection:
                roi = image
                bbox = None
            else:
                return None, None
        
        # Resize to target size
        roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
        
        return roi, bbox
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/detect",
        name: str = "lens_detector"
    ) -> str:
        """
        Train the YOLO model on lens detection dataset.
        
        Args:
            data_yaml: Path to YOLO data configuration file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory for saving results
            name: Experiment name
            
        Returns:
            Path to best model weights
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=self.device,
            verbose=True
        )
        
        # Return path to best weights
        best_path = Path(project) / name / "weights" / "best.pt"
        return str(best_path)
    
    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        simplify: bool = True
    ) -> str:
        """
        Export model to different formats for deployment.
        
        Args:
            format: Export format ('onnx', 'tflite', 'engine', etc.)
            imgsz: Image size for export
            simplify: Whether to simplify ONNX model
            
        Returns:
            Path to exported model
        """
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            simplify=simplify
        )
        return export_path


class LensDetectorDataset:
    """
    Helper class to create YOLO-format dataset for lens detection.
    
    YOLO format:
        - images/train/*.jpg
        - images/val/*.jpg
        - labels/train/*.txt  (class x_center y_center width height)
        - labels/val/*.txt
        - data.yaml
    """
    
    @staticmethod
    def create_data_yaml(
        output_dir: str,
        train_images: str = "images/train",
        val_images: str = "images/val",
        class_names: List[str] = ["lens"]
    ) -> str:
        """
        Create YOLO data configuration file.
        
        Args:
            output_dir: Directory to save data.yaml
            train_images: Path to training images (relative to output_dir)
            val_images: Path to validation images (relative to output_dir)
            class_names: List of class names
            
        Returns:
            Path to created data.yaml
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_content = f"""# Lens Detection Dataset
path: {output_dir.absolute()}
train: {train_images}
val: {val_images}

# Classes
names:
"""
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return str(yaml_path)
    
    @staticmethod
    def convert_bbox_to_yolo(
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height].
        All values are normalized to [0, 1].
        """
        x1, y1, x2, y2 = bbox
        
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return x_center, y_center, width, height


# For testing
if __name__ == "__main__":
    print("Testing LensDetector...")
    
    # Create detector (using pretrained YOLOv8n)
    detector = LensDetector(conf_threshold=0.5)
    print(f"Model loaded. Trained for lens detection: {detector.is_trained}")
    print(f"Device: {detector.device}")
    
    # Create a dummy test image
    dummy_image = np.random.randint(0, 255, (2448, 3264, 3), dtype=np.uint8)
    
    # Test detection (will likely not detect anything on random noise)
    bbox = detector.detect(dummy_image)
    print(f"Detection result: {bbox}")
    
    # Test crop
    roi, bbox = detector.detect_and_crop(
        dummy_image,
        target_size=(384, 384),
        return_original_if_no_detection=True
    )
    print(f"ROI shape: {roi.shape}")
    
    print("\nTo train for lens detection, you need to:")
    print("1. Annotate lens regions in slit-lamp images using LabelImg or Roboflow")
    print("2. Create data.yaml using LensDetectorDataset.create_data_yaml()")
    print("3. Call detector.train(data_yaml='path/to/data.yaml')")
