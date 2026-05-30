"""
Explainable AI (XAI) Tools for Cataract Detection Baseline.

Includes:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- LIME: Local Interpretable Model-agnostic Explanations  
- SHAP: SHapley Additive exPlanations

Usage:
    python baseline_xai/evaluation/xai_tools.py --model_path checkpoints/best_baseline.pth --image path/to/image.jpg
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============ GRAD-CAM ============

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Shows which regions of the image the model focuses on for its prediction.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute CAM for (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Default: use the last conv layer of MobileNetV2
        if target_layer is None:
            self.target_layer = model.backbone.features[-1]
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None, task='binary'):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class (None = predicted class)
            task: 'binary' or 'severity'
            
        Returns:
            cam: Heatmap as numpy array (H, W)
        """
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if task == 'binary':
            logits = output['binary_logits']
        else:
            logits = output['severity_logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()
    
    def visualize(self, image, cam, alpha=0.4, save_path=None):
        """
        Overlay CAM on original image.
        
        Args:
            image: Original PIL Image
            cam: CAM heatmap
            alpha: Overlay transparency
            save_path: Path to save visualization
        """
        # Convert image to numpy
        img_array = np.array(image.resize((224, 224)))
        
        # Resize CAM to match image
        import cv2
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Create heatmap
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay
        overlay = (1 - alpha) * img_array + alpha * heatmap
        overlay = overlay.astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return overlay


# ============ LIME ============

def lime_explain(model, image, device='cuda', num_samples=1000):
    """
    Generate LIME explanation for image classification.
    
    Args:
        model: PyTorch model
        image: PIL Image
        device: Device to run on
        num_samples: Number of perturbation samples
        
    Returns:
        explanation: LIME explanation object
    """
    try:
        from lime import lime_image
    except ImportError:
        print("LIME not installed. Run: pip install lime")
        return None
    
    # Preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def predict_fn(images):
        """Prediction function for LIME."""
        batch = torch.stack([preprocess(Image.fromarray(img)) for img in images])
        batch = batch.to(device)
        
        with torch.no_grad():
            output = model(batch)
            probs = F.softmax(output['binary_logits'], dim=1)
        
        return probs.cpu().numpy()
    
    model.eval()
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Convert image to numpy
    img_array = np.array(image.resize((224, 224)))
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples
    )
    
    return explanation


def visualize_lime(explanation, image, label=1, save_path=None):
    """
    Visualize LIME explanation.
    
    Args:
        explanation: LIME explanation object
        image: Original PIL Image
        label: Class label to explain
        save_path: Path to save visualization
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    
    # Get image and mask
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Get boundaries
    img_array = np.array(image.resize((224, 224)))
    img_boundary = mark_boundaries(temp / 255.0, mask)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='RdBu_r')
    axes[1].set_title('LIME Regions')
    axes[1].axis('off')
    
    axes[2].imshow(img_boundary)
    axes[2].set_title('Important Features')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ============ SHAP ============

def shap_explain(model, image, background_images, device='cuda'):
    """
    Generate SHAP explanation for image.
    
    Args:
        model: PyTorch model
        image: PIL Image to explain
        background_images: List of background PIL Images for SHAP
        device: Device to run on
        
    Returns:
        shap_values: SHAP values
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Prepare background
    background = torch.stack([preprocess(img) for img in background_images])
    background = background.to(device)
    
    # Prepare input
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Create wrapper for SHAP
    def model_predict(x):
        output = model(x)
        return output['binary_logits']
    
    model.eval()
    
    # Create explainer
    explainer = shap.DeepExplainer(model_predict, background)
    
    # Get SHAP values
    shap_values = explainer.shap_values(input_tensor)
    
    return shap_values


def visualize_shap(shap_values, image, save_path=None):
    """
    Visualize SHAP values.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed")
        return
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img_tensor = preprocess(image).numpy().transpose(1, 2, 0)
    
    # Plot
    shap.image_plot(shap_values, img_tensor)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")


# ============ COMBINED XAI ANALYSIS ============

def full_xai_analysis(model, image_path, device='cuda', output_dir='baseline_xai/xai_outputs'):
    """
    Run full XAI analysis on a single image.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image
        device: Device to run on
        output_dir: Directory to save outputs
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        binary_prob = F.softmax(output['binary_logits'], dim=1)
        severity_prob = F.softmax(output['severity_logits'], dim=1)
    
    binary_pred = binary_prob.argmax(dim=1).item()
    severity_pred = severity_prob.argmax(dim=1).item()
    
    print(f"\n{'='*50}")
    print(f"XAI Analysis: {Path(image_path).name}")
    print(f"{'='*50}")
    print(f"Prediction: {'Cataract' if binary_pred == 1 else 'Normal'} ({binary_prob[0, binary_pred]:.2%})")
    print(f"Severity: LOCS {severity_pred} ({severity_prob[0, severity_pred]:.2%})")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Grad-CAM
    print("\n[1/3] Generating Grad-CAM...")
    gradcam = GradCAM(model)
    cam = gradcam.generate(input_tensor, task='binary')
    gradcam.visualize(image, cam, save_path=output_dir / f'{Path(image_path).stem}_gradcam.png')
    
    # 2. LIME
    print("\n[2/3] Generating LIME explanation...")
    try:
        explanation = lime_explain(model, image, device=device, num_samples=500)
        if explanation:
            visualize_lime(explanation, image, label=binary_pred, save_path=output_dir / f'{Path(image_path).stem}_lime.png')
    except Exception as e:
        print(f"LIME failed: {e}")
    
    # 3. Summary
    print("\n[3/3] XAI Analysis Complete!")
    print(f"Outputs saved to: {output_dir}")
    
    return {
        'binary_pred': binary_pred,
        'binary_prob': binary_prob.cpu().numpy(),
        'severity_pred': severity_pred,
        'severity_prob': severity_prob.cpu().numpy(),
        'cam': cam
    }


# ============ MAIN ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XAI Analysis for Cataract Detection")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image to analyze')
    parser.add_argument('--output_dir', type=str, default='baseline_xai/xai_outputs', help='Output directory')
    args = parser.parse_args()
    
    # Load model
    from models.mobilenet_baseline import MobileNetBaseline
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MobileNetBaseline()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from: {args.model_path}")
    
    # Run analysis
    full_xai_analysis(model, args.image, device=device, output_dir=args.output_dir)
