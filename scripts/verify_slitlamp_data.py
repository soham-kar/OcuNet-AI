"""
Checkpoint 1: Verify Slit-lamp Dataset and LOCS Annotations

This script checks:
1. Excel file with LOCS annotations exists
2. Image files are accessible
3. Class distribution for LOCS grades
4. Computes class weights for training
"""

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def verify_slitlamp_data():
    print("=" * 60)
    print("CHECKPOINT 1: Slit-lamp Data Verification")
    print("=" * 60)
    
    # Path to LOCS annotations
    base_dir = Path("data/raw/slitlamp/Nuclear Cataract Database for Biomedical and Machine Learning Applications/Nuclear Cataract Dataset")
    excel_path = base_dir / "Base de Datos LOCS pacientes LLENO.xlsx"
    
    # Check if Excel exists
    if not excel_path.exists():
        print(f"❌ Excel file not found at: {excel_path}")
        print("\nSearching for alternative Excel files...")
        excel_files = [f for f in base_dir.glob("*.xlsx") if not f.name.startswith('~$')]
        if excel_files:
            print(f"Found {len(excel_files)} Excel files:")
            for f in excel_files:
                print(f"  - {f.name}")
            excel_path = excel_files[0]
            print(f"\nUsing: {excel_path.name}")
        else:
            print("No Excel files found!")
            return
    
    # Load Excel
    print(f"\n✅ Loading: {excel_path.name}")
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        print("\nTrying alternative engines...")
        try:
            df = pd.read_excel(excel_path, engine='xlrd')
        except:
            print("❌ Failed with all engines. Please check the file format.")
            return
    
    # Display columns
    print(f"\nColumns found: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    # Try to identify LOCS column
    locs_col = None
    for col in df.columns:
        if 'locs' in col.lower() or 'grade' in col.lower():
            locs_col = col
            break
    
    if locs_col is None:
        print("\n⚠️ Could not auto-detect LOCS column")
        print("Please manually specify the column name containing LOCS grades (0-6)")
        return
    
    print(f"\n✅ Using LOCS column: '{locs_col}'")
    
    # Class distribution
    print("\n" + "=" * 60)
    print("LOCS Grade Distribution:")
    print("=" * 60)
    locs_counts = df[locs_col].value_counts().sort_index()
    for grade, count in locs_counts.items():
        pct = count / len(df) * 100
        print(f"  LOCS {grade}: {count:4d} images ({pct:5.1f}%)")
    
    # Missing values
    missing = df[locs_col].isna().sum()
    if missing > 0:
        print(f"\n⚠️ Missing LOCS grades: {missing} ({missing/len(df)*100:.1f}%)")
    else:
        print(f"\n✅ No missing LOCS grades")
    
    # Compute class weights
    print("\n" + "=" * 60)
    print("Class Weights for Balanced Training:")
    print("=" * 60)
    
    locs_grades = df[locs_col].dropna().values
    unique_grades = np.unique(locs_grades)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_grades,
        y=locs_grades
    )
    
    for grade, weight in zip(unique_grades, class_weights):
        print(f"  LOCS {int(grade)}: {weight:.3f}")
    
    # Check for image paths
    print("\n" + "=" * 60)
    print("Image File Verification:")
    print("=" * 60)
    
    # Try to find image column
    img_col = None
    for col in df.columns:
        if 'image' in col.lower() or 'file' in col.lower() or 'path' in col.lower():
            img_col = col
            break
    
    if img_col:
        print(f"✅ Image column found: '{img_col}'")
        print(f"Sample paths:")
        for path in df[img_col].head(3):
            print(f"  - {path}")
    else:
        print("⚠️ Could not auto-detect image path column")
    
    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 1 SUMMARY:")
    print("=" * 60)
    print(f"✅ Total images: {len(df)}")
    print(f"✅ LOCS grades: {len(unique_grades)} classes ({unique_grades.min():.0f}-{unique_grades.max():.0f})")
    print(f"✅ Class weights computed")
    
    if missing == 0 and img_col:
        print("\n🎉 Dataset ready for training!")
    else:
        print("\n⚠️ Some issues need attention before training")
    
    # Save processed labels
    output_path = Path("data/raw/slitlamp/labels_mendeley.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create standardized CSV
    processed_df = pd.DataFrame({
        'image_path': df[img_col] if img_col else df.iloc[:, 0],
        'locs_grade': df[locs_col]
    })
    
    processed_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved processed labels to: {output_path}")


if __name__ == "__main__":
    verify_slitlamp_data()
