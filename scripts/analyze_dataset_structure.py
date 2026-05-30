"""
Checkpoint 1 Alternative: Dataset Structure Analysis
This script analyzes the slit-lamp dataset structure without requiring the Excel file
"""

from pathlib import Path
import json

def analyze_dataset_structure():
    print("=" * 60)
    print("CHECKPOINT 1 (Alternative): Dataset Structure Analysis")
    print("=" * 60)
    
    base_dir = Path("data/raw/slitlamp/Nuclear Cataract Database for Biomedical and Machine Learning Applications/Nuclear Cataract Dataset")
    
    if not base_dir.exists():
        print(f"❌ Dataset directory not found: {base_dir}")
        return
    
    # Find all patient folders
    patient_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print(f"\n✅ Found {len(patient_folders)} patient folders")
    print(f"   Range: {patient_folders[0].name} to {patient_folders[-1].name}")
    
    # Analyze structure
    total_images = 0
    eyes_data = {'DER': 0, 'IZQ': 0}
    sample_structure = {}
    
    for i, patient_dir in enumerate(patient_folders[:5]):  # Sample first 5
        patient_id = patient_dir.name
        sample_structure[patient_id] = {}
        
        for eye_dir in patient_dir.iterdir():
            if eye_dir.is_dir():
                eye = eye_dir.name
                images = list(eye_dir.glob("*.JPG")) + list(eye_dir.glob("*.jpg"))
                sample_structure[patient_id][eye] = len(images)
                
                if i == 0:  # Full count only for first patient
                    eyes_data[eye] = eyes_data.get(eye, 0) + len(images)
    
    # Count all images
    for patient_dir in patient_folders:
        for eye_dir in patient_dir.iterdir():
            if eye_dir.is_dir():
                images = list(eye_dir.glob("*.JPG")) + list(eye_dir.glob("*.jpg"))
                total_images += len(images)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total patients: {len(patient_folders)}")
    print(f"   Total images: {total_images}")
    print(f"   Avg images per patient: {total_images / len(patient_folders):.1f}")
    
    print(f"\n📁 Sample Structure (first 5 patients):")
    for patient_id, eyes in sample_structure.items():
        print(f"   Patient {patient_id}:")
        for eye, count in eyes.items():
            print(f"      {eye}: {count} images")
    
    print(f"\n⚠️  LOCS Annotations:")
    print(f"   Excel file is currently OPEN")
    print(f"   File: 'Base de Datos LOCS pacientes LLENO.xlsx'")
    print(f"   Action required: Close the Excel file to proceed")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Close the Excel file")
    print(f"   2. Run: python scripts/verify_slitlamp_data.py")
    print(f"   3. This will parse LOCS grades and create labels_mendeley.csv")
    
    # Save structure info
    output = {
        'num_patients': len(patient_folders),
        'total_images': total_images,
        'avg_images_per_patient': total_images / len(patient_folders),
        'patient_range': f"{patient_folders[0].name}-{patient_folders[-1].name}",
        'excel_status': 'OPEN (temp file detected)',
        'action_required': 'Close Excel file to access LOCS annotations'
    }
    
    output_path = Path("data/raw/slitlamp/dataset_structure.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Structure info saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    result = analyze_dataset_structure()
    
    if result:
        print("\n" + "=" * 60)
        print("CHECKPOINT 1 STATUS: ⏸️  PAUSED")
        print("=" * 60)
        print("Waiting for Excel file to be closed...")
        print("Once closed, re-run: python scripts/verify_slitlamp_data.py")
