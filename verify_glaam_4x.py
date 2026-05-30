"""
Verify GLAAM-4X Results - Compare with Baseline and Previous GLAAM
"""

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, roc_curve

DISEASE_NAMES = ['DR', 'Glaucoma', 'Cataract', 'Myopia']

def compute_all_metrics(predictions, labels):
    """Compute all metrics for verification."""
    results = {}
    
    for i, disease in enumerate(DISEASE_NAMES):
        y_true = labels[:, i]
        y_prob = predictions[:, i]
        y_pred = (y_prob > 0.5).astype(int)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            bacc = balanced_accuracy_score(y_true, y_pred)
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            idx = np.where(fpr <= 0.05)[0]
            sens_95 = tpr[idx[-1]] if len(idx) > 0 else 0.0
            
            results[disease] = {
                'AUC': auc,
                'PR-AUC': pr_auc,
                'BalAcc': bacc,
                'Sens@95%Spec': sens_95
            }
    
    # Compute mean
    mean_auc = np.mean([results[d]['AUC'] for d in DISEASE_NAMES if d in results])
    mean_pr_auc = np.mean([results[d]['PR-AUC'] for d in DISEASE_NAMES if d in results])
    results['Mean'] = {'AUC': mean_auc, 'PR-AUC': mean_pr_auc}
    
    return results

def main():
    print("="*70)
    print("🔬 GLAAM-4X VERIFICATION: Disease-Specific Attention Results")
    print("="*70)
    
    # Load GLAAM-4X predictions
    print("\n📁 Loading GLAAM-4X predictions...")
    try:
        with open("checkpoints_glaam/glaam_4x_test_predictions.pkl", "rb") as f:
            glaam_4x_data = pickle.load(f)
        glaam_4x_preds = glaam_4x_data['predictions']
        glaam_4x_labels = glaam_4x_data['labels']
        print(f"   GLAAM-4X: {len(glaam_4x_preds)} samples")
    except Exception as e:
        print(f"   ⚠️ Could not load GLAAM-4X: {e}")
        return
    
    # Load baseline predictions
    print("\n📁 Loading Baseline predictions...")
    try:
        with open("checkpoints_glaam/baseline_ultra_fixed_predictions.pkl", "rb") as f:
            baseline_data = pickle.load(f)
        baseline_preds = baseline_data['predictions']
        baseline_labels = baseline_data['labels']
        print(f"   Baseline: {len(baseline_preds)} samples")
    except:
        baseline_preds = None
    
    # Load previous GLAAM predictions
    print("\n📁 Loading Previous GLAAM predictions...")
    try:
        with open("checkpoints_glaam/glaam_final_predictions.pkl", "rb") as f:
            glaam_data = pickle.load(f)
        glaam_preds = glaam_data['predictions']
        glaam_labels = glaam_data['labels']
        print(f"   GLAAM: {len(glaam_preds)} samples")
    except:
        glaam_preds = None
    
    # Compute metrics
    print("\n🔢 Computing metrics...")
    glaam_4x_results = compute_all_metrics(glaam_4x_preds, glaam_4x_labels)
    
    if baseline_preds is not None:
        baseline_results = compute_all_metrics(baseline_preds, baseline_labels)
    if glaam_preds is not None:
        glaam_results = compute_all_metrics(glaam_preds, glaam_labels)
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 PER-DISEASE AUC COMPARISON")
    print("="*80)
    print(f"\n{'Disease':<12} {'Baseline':<12} {'GLAAM':<12} {'GLAAM-4X':<12} {'Δ vs Base':<12}")
    print("-"*60)
    
    for disease in DISEASE_NAMES:
        base_auc = baseline_results.get(disease, {}).get('AUC', 0) if baseline_preds is not None else 0
        glaam_auc = glaam_results.get(disease, {}).get('AUC', 0) if glaam_preds is not None else 0
        glaam_4x_auc = glaam_4x_results.get(disease, {}).get('AUC', 0)
        
        delta = (glaam_4x_auc - base_auc) * 100 if base_auc > 0 else 0
        print(f"{disease:<12} {base_auc:<12.4f} {glaam_auc:<12.4f} {glaam_4x_auc:<12.4f} {delta:+.2f}%")
    
    # Mean
    print("-"*60)
    base_mean = baseline_results.get('Mean', {}).get('AUC', 0) if baseline_preds is not None else 0
    glaam_mean = glaam_results.get('Mean', {}).get('AUC', 0) if glaam_preds is not None else 0
    glaam_4x_mean = glaam_4x_results.get('Mean', {}).get('AUC', 0)
    delta_mean = (glaam_4x_mean - base_mean) * 100 if base_mean > 0 else 0
    print(f"{'Mean':<12} {base_mean:<12.4f} {glaam_mean:<12.4f} {glaam_4x_mean:<12.4f} {delta_mean:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
