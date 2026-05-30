"""
Verify GLAAM vs Baseline Results
Loads saved predictions and computes metrics independently to confirm reported results.
"""

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, roc_curve

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

def compute_all_metrics(predictions, labels):
    """Compute all metrics for verification."""
    results = {}
    
    class_aucs = []
    class_pr_aucs = []
    class_balanced_accs = []
    class_sens_at_95_spec = []
    
    for i, disease in enumerate(DISEASE_NAMES):
        y_true = labels[:, i]
        y_prob = predictions[:, i]
        y_pred = (y_prob > 0.5).astype(int)
        
        if len(np.unique(y_true)) > 1:
            # ROC-AUC
            auc = roc_auc_score(y_true, y_prob)
            class_aucs.append(auc)
            
            # PR-AUC
            pr_auc = average_precision_score(y_true, y_prob)
            class_pr_aucs.append(pr_auc)
            
            # Balanced Accuracy
            bacc = balanced_accuracy_score(y_true, y_pred)
            class_balanced_accs.append(bacc)
            
            # Sensitivity @ 95% Specificity
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            idx = np.where(fpr <= 0.05)[0]
            if len(idx) > 0:
                sens_95 = tpr[idx[-1]]
            else:
                sens_95 = 0.0
            class_sens_at_95_spec.append(sens_95)
            
            results[disease] = {
                'AUC': auc,
                'PR-AUC': pr_auc,
                'BalAcc': bacc,
                'Sens@95%Spec': sens_95
            }
    
    results['Mean'] = {
        'AUC': np.mean(class_aucs),
        'PR-AUC': np.mean(class_pr_aucs),
        'BalAcc': np.mean(class_balanced_accs),
        'Sens@95%Spec': np.mean(class_sens_at_95_spec)
    }
    
    return results

def print_comparison_table(baseline_results, glaam_results):
    """Print a comparison table."""
    print("\n" + "="*80)
    print("📊 VERIFIED RESULTS COMPARISON: BASELINE vs GLAAM")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Baseline':<12} {'GLAAM':<12} {'Improvement':<15}")
    print("-"*60)
    
    for metric in ['AUC', 'PR-AUC', 'BalAcc', 'Sens@95%Spec']:
        baseline_val = baseline_results['Mean'][metric]
        glaam_val = glaam_results['Mean'][metric]
        improvement = (glaam_val - baseline_val) / baseline_val * 100
        
        print(f"Mean {metric:<15} {baseline_val:<12.4f} {glaam_val:<12.4f} {improvement:+.2f}%")
    
    print("\n" + "-"*80)
    print("PER-DISEASE AUC COMPARISON:")
    print("-"*80)
    print(f"{'Disease':<15} {'Baseline AUC':<15} {'GLAAM AUC':<15} {'Improvement':<15}")
    print("-"*60)
    
    for disease in DISEASE_NAMES:
        baseline_auc = baseline_results[disease]['AUC']
        glaam_auc = glaam_results[disease]['AUC']
        improvement = (glaam_auc - baseline_auc) / baseline_auc * 100
        print(f"{disease:<15} {baseline_auc:<15.4f} {glaam_auc:<15.4f} {improvement:+.2f}%")

def main():
    print("🔍 VERIFICATION SCRIPT: Checking GLAAM vs Baseline Results")
    print("="*60)
    
    # Load GLAAM predictions
    print("\n📁 Loading GLAAM predictions...")
    with open("checkpoints_glaam/glaam_final_predictions.pkl", "rb") as f:
        glaam_data = pickle.load(f)
    
    glaam_preds = glaam_data['predictions']
    glaam_labels = glaam_data['labels']
    print(f"   GLAAM: {len(glaam_preds)} samples")
    
    # Load Baseline predictions
    print("\n📁 Loading Baseline predictions...")
    with open("checkpoints_glaam/baseline_ultra_fixed_predictions.pkl", "rb") as f:
        baseline_data = pickle.load(f)
    
    baseline_preds = baseline_data['predictions']
    baseline_labels = baseline_data['labels']
    print(f"   Baseline: {len(baseline_preds)} samples")
    
    # Compute metrics
    print("\n🔢 Computing metrics...")
    baseline_results = compute_all_metrics(baseline_preds, baseline_labels)
    glaam_results = compute_all_metrics(glaam_preds, glaam_labels)
    
    # Print comparison
    print_comparison_table(baseline_results, glaam_results)
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
