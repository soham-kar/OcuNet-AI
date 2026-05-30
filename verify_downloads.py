import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Move attention maps if needed
import shutil
src = Path("glaam_odir_v1_attention_maps.pkl")
if src.exists():
    shutil.move(str(src), "checkpoints_glaam/glaam_final_attention_maps.pkl")
    print("Moved attention maps to checkpoints_glaam/")

# Load and verify predictions
pred_path = "checkpoints_glaam/glaam_final_predictions.pkl"
d = pickle.load(open(pred_path, 'rb'))

print("=" * 60)
print("PREDICTIONS VERIFICATION")
print("=" * 60)
print(f"Keys      : {list(d.keys())}")
print(f"Samples   : {len(d['predictions'])}")
print(f"Diseases  : {d['disease_names']}")
print(f"Has logits: {'logits' in d}")

if 'logits' in d:
    print(f"Logits range: {d['logits'].min():.2f} to {d['logits'].max():.2f}")

print("\nPer-disease AUC:")
for i, name in enumerate(d['disease_names']):
    auc = roc_auc_score(d['labels'][:, i], d['predictions'][:, i])
    print(f"  {name}: {auc:.4f}")

mean_auc = np.mean([
    roc_auc_score(d['labels'][:, i], d['predictions'][:, i])
    for i in range(len(d['disease_names']))
])
print(f"  Mean AUC: {mean_auc:.4f}")

# Verify attention maps
attn_path = "checkpoints_glaam/glaam_final_attention_maps.pkl"
if Path(attn_path).exists():
    attn = pickle.load(open(attn_path, 'rb'))
    print("\n" + "=" * 60)
    print("ATTENTION MAPS VERIFICATION")
    print("=" * 60)
    print(f"Keys    : {list(attn.keys())}")
    print(f"Samples : {len(attn['paths'])}")
    print("Sample labels:")
    for i, (path, label) in enumerate(zip(attn['paths'], attn['labels'])):
        fname = Path(path).name
        disease_names = d['disease_names']
        active = [disease_names[j] for j in range(len(disease_names)) if label[j] == 1] or ['Normal']
        print(f"  [{i+1}] {fname} -> {active}")
