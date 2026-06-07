"""
Create Phase 1 (v3) data splits for ASL training.

Changes from v2:
  - val_combined.csv → test_v3.csv (1,407 images, NEVER used for tuning)
  - train_combined.csv → train_v3.csv (7,170) + val_tune_v3.csv (797)
  - val_tune_v3.csv used ONLY for threshold optimization during training

Usage:
    python create_v3_splits.py

Outputs:
    data/train_v3.csv       — 7,170 images (training)
    data/val_tune_v3.csv    — 797 images (threshold tuning)
    data/test_v3.csv        — 1,407 images (final evaluation, never touched)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load original splits
train_df = pd.read_csv("data/train_combined.csv")
val_df = pd.read_csv("data/val_combined.csv")

print(f"Original train: {len(train_df)} images")
print(f"Original val:   {len(val_df)} images")

# val_combined.csv becomes the dedicated test set
test_df = val_df.copy()
test_df.to_csv("data/test_v3.csv", index=False)
print(f"\n✅ test_v3.csv: {len(test_df)} images (dedicated test set)")

# Split original train into new train (90%) + val_tune (10%)
train_df['n_diseases'] = train_df[['Cataract', 'DR', 'Glaucoma', 'Myopia']].sum(axis=1)
train_df['strat_group'] = train_df['n_diseases'].clip(upper=2).astype(str)

train_new, val_tune = train_test_split(
    train_df,
    test_size=0.10,
    stratify=train_df['strat_group'],
    random_state=42,
)

# Drop helper columns
train_new = train_new.drop(columns=['n_diseases', 'strat_group'])
val_tune = val_tune.drop(columns=['n_diseases', 'strat_group'])

# Save
train_new.to_csv("data/train_v3.csv", index=False)
val_tune.to_csv("data/val_tune_v3.csv", index=False)

print(f"✅ train_v3.csv:    {len(train_new)} images (training)")
print(f"✅ val_tune_v3.csv: {len(val_tune)} images (threshold tuning only)")

# Print distributions
print("\n" + "=" * 60)
print("Disease Distribution Comparison")
print("=" * 60)
for name, df in [("train_v3", train_new), ("val_tune_v3", val_tune), ("test_v3", test_df)]:
    print(f"\n{name} ({len(df)} images):")
    for d in ['Cataract', 'DR', 'Glaucoma', 'Myopia']:
        pos = df[d].sum()
        print(f"  {d:12s}: {int(pos):5d} ({pos/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ All splits created. Ready for v3 ASL training.")
print("=" * 60)
