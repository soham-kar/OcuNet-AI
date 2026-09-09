"""Quick test: load model, run inference on a sample image."""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from models.glaam_4x import GLAAM_4X

DISEASES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

class GLAAM4XClassifier(nn.Module):
    REORDER_IDX = [2, 0, 1, 3]
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = GLAAM_4X(pretrained=False, dropout_rate=dropout_rate)
    def forward(self, x):
        return self.backbone(x)['logits'][:, self.REORDER_IDX]

print("Loading model...")
model = GLAAM4XClassifier()
sd = torch.load('model_weights.pth', map_location='cpu', weights_only=True)
# Strip _orig_mod. prefix from torch.compile()
sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print("✅ Model loaded")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

for sample in ['cataract_sample.jpg', 'dr_sample.jpg', 'glaucoma_sample.jpg', 'myopia_sample.jpg', 'normal_sample.jpg']:
    img = Image.open(f'demo_images/{sample}').convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor).numpy()[0]
    probs = 1 / (1 + np.exp(-logits))
    print(f"\n{sample}:")
    for d, p in zip(DISEASES, probs):
        print(f"  {d:12s}: {p:.1%}")