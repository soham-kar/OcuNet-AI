"""
Download PALM dataset from Hugging Face.
Only downloads Training/ and Validation/ folders (~4.2 GB instead of 9.37 GB).

Usage:
    python download_palm.py

Output:
    data/raw/PALM/
    ├── Training/
    │   ├── Classification/   (images)
    │   └── Annotation/       (labels)
    └── Validation/
        ├── Classification/
        └── Annotation/
"""

from huggingface_hub import snapshot_download
from pathlib import Path

# Target directory inside the project
LOCAL_DIR = Path(__file__).parent / "data" / "raw" / "PALM"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Downloading PALM dataset from Hugging Face")
print("=" * 60)
print(f"Target: {LOCAL_DIR}")
print("Only downloading Training/ and Validation/ folders")
print("(skipping Test/, rm_images/, etc.)")
print()

snapshot_download(
    repo_id="ctmedtech/PALM",
    repo_type="dataset",
    local_dir=str(LOCAL_DIR),
    allow_patterns=[
        "Training/Classification/*",
        "Training/Annotation/*",
        "Validation/Classification/*",
        "Validation/Annotation/*",
    ],
)

print()
print("=" * 60)
print("Download complete!")
print("=" * 60)

# Verify structure
for sub in ["Training/Classification", "Training/Annotation",
            "Validation/Classification", "Validation/Annotation"]:
    p = LOCAL_DIR / sub
    n = len(list(p.glob("*"))) if p.exists() else 0
    print(f"  {sub}: {n} files")
