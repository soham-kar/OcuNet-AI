"""
Create train/validation split from the unified dataset.
Stratifies by multi-label presence to preserve rare multi-label cases.

Usage:
    python create_train_val_split.py

Outputs:
    data/train_combined.csv
    data/val_combined.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load unified dataset
df = pd.read_csv("data/combined_fundus_dataset.csv")
print(f"Loaded {len(df)} images")

# Create stratification flag: 0 = no disease, 1 = single disease, 2+ = multi-label
df['n_diseases'] = df[['Cataract', 'DR', 'Glaucoma', 'Myopia']].sum(axis=1)
df['strat_group'] = df['n_diseases'].clip(upper=2).astype(str)  # '0', '1', '2'

# Split: 85% train, 15% val
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df['strat_group'],
    random_state=42
)

# Drop helper columns
train_df = train_df.drop(columns=['n_diseases', 'strat_group'])
val_df = val_df.drop(columns=['n_diseases', 'strat_group'])

# Save
train_df.to_csv("data/train_combined.csv", index=False)
val_df.to_csv("data/val_combined.csv", index=False)

print(f"\nTrain: {len(train_df)} images")
print(f"Val:   {len(val_df)} images")
print()
print("Train disease distribution:")
for d in ['Cataract', 'DR', 'Glaucoma', 'Myopia']:
    pos = train_df[d].sum()
    print(f"  {d}: {pos} ({pos/len(train_df)*100:.1f}%)")
print()
print("Val disease distribution:")
for d in ['Cataract', 'DR', 'Glaucoma', 'Myopia']:
    pos = val_df[d].sum()
    print(f"  {d}: {pos} ({pos/len(val_df)*100:.1f}%)")
print()
print("Saved to data/train_combined.csv and data/val_combined.csv")
