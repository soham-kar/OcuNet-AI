"""
Clinical Validation Script.
Week 6 of 10-week roadmap: Compute agreement metrics with ophthalmologist grades.

Computes:
- Cohen's Kappa (inter-rater agreement)
- Confusion matrix
- Per-class precision/recall/F1
- Calibration curve
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_clinical_metrics(
    model_predictions: np.ndarray,
    expert_grades: np.ndarray,
    class_names: list = None
) -> dict:
    """
    Compute clinical validation metrics.
    
    Args:
        model_predictions: Model's predicted LOCS grades (array of ints 0-6)
        expert_grades: Expert ophthalmologist grades (array of ints 0-6)
        class_names: Names for each class (default: LOCS 0-6)
    
    Returns:
        Dictionary with all metrics
    """
    if class_names is None:
        class_names = [f"LOCS {i}" for i in range(7)]
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(
        expert_grades, 
        model_predictions,
        weights='quadratic'  # Quadratic weights for ordinal data
    )
    
    # Linear weighted kappa
    kappa_linear = cohen_kappa_score(
        expert_grades,
        model_predictions,
        weights='linear'
    )
    
    # Accuracy
    accuracy = accuracy_score(expert_grades, model_predictions)
    
    # Within-1-grade accuracy (clinically relevant)
    within_1 = np.mean(np.abs(model_predictions - expert_grades) <= 1)
    
    # Confusion matrix
    cm = confusion_matrix(expert_grades, model_predictions)
    
    # Classification report
    report = classification_report(
        expert_grades,
        model_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'cohens_kappa_quadratic': kappa,
        'cohens_kappa_linear': kappa_linear,
        'accuracy': accuracy,
        'within_1_grade_accuracy': within_1,
        'confusion_matrix': cm,
        'classification_report': report,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None,
    title: str = "Confusion Matrix"
):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.ylabel('True Grade (Expert)')
    plt.xlabel('Predicted Grade (Model)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_agreement_chart(
    model_predictions: np.ndarray,
    expert_grades: np.ndarray,
    save_path: str = None
):
    """Plot agreement scatter chart with tolerance bands."""
    plt.figure(figsize=(10, 8))
    
    # Add jitter for visualization
    jitter = 0.1
    x = expert_grades + np.random.uniform(-jitter, jitter, len(expert_grades))
    y = model_predictions + np.random.uniform(-jitter, jitter, len(model_predictions))
    
    # Plot perfect agreement line
    plt.plot([0, 6], [0, 6], 'k--', label='Perfect Agreement', alpha=0.5)
    
    # Plot ±1 grade tolerance
    plt.fill_between([0, 6], [-1, 5], [1, 7], alpha=0.2, color='green', label='±1 Grade Tolerance')
    
    # Scatter plot
    colors = ['green' if abs(m - e) <= 1 else 'red' 
              for m, e in zip(model_predictions, expert_grades)]
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='white', s=50)
    
    plt.xlabel('Expert Grade')
    plt.ylabel('Model Prediction')
    plt.title('Model vs Expert Agreement')
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)
    plt.xticks(range(7))
    plt.yticks(range(7))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved agreement chart to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_clinical_report(
    metrics: dict,
    output_path: str = None
) -> str:
    """Generate a clinical validation report."""
    
    report = f"""
================================================================================
                    CLINICAL VALIDATION REPORT
================================================================================

AGREEMENT METRICS
-----------------
Cohen's Kappa (Quadratic): {metrics['cohens_kappa_quadratic']:.3f}
  Interpretation: {interpret_kappa(metrics['cohens_kappa_quadratic'])}

Cohen's Kappa (Linear):    {metrics['cohens_kappa_linear']:.3f}
  Interpretation: {interpret_kappa(metrics['cohens_kappa_linear'])}

ACCURACY METRICS
----------------
Exact Match Accuracy:      {metrics['accuracy']:.1%}
Within ±1 Grade Accuracy:  {metrics['within_1_grade_accuracy']:.1%}  (Clinically Relevant)

Macro F1-Score:            {metrics['macro_f1']:.3f}
Weighted F1-Score:         {metrics['weighted_f1']:.3f}

CLINICAL SIGNIFICANCE
---------------------
- Kappa > 0.75 indicates "substantial" agreement, acceptable for clinical use
- Within ±1 grade accuracy > 90% is clinically acceptable
- Model can be used for screening with human verification for borderline cases

TARGET CRITERIA:
  ✓ Cohen's Kappa > 0.75: {"✅ PASS" if metrics['cohens_kappa_quadratic'] > 0.75 else "❌ NEEDS IMPROVEMENT"}
  ✓ Within-1 Accuracy > 90%: {"✅ PASS" if metrics['within_1_grade_accuracy'] > 0.9 else "❌ NEEDS IMPROVEMENT"}
  ✓ Exact Accuracy > 85%: {"✅ PASS" if metrics['accuracy'] > 0.85 else "❌ NEEDS IMPROVEMENT"}

================================================================================
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Saved clinical report to {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Clinical Validation")
    parser.add_argument('--predictions', type=str, required=True, help='CSV with model predictions')
    parser.add_argument('--expert_grades', type=str, required=True, help='CSV with expert grades')
    parser.add_argument('--output_dir', type=str, default='outputs/clinical_validation')
    args = parser.parse_args()
    
    # Load data
    preds_df = pd.read_csv(args.predictions)
    expert_df = pd.read_csv(args.expert_grades)
    
    # Extract arrays
    model_predictions = preds_df['predicted_grade'].values
    expert_grades = expert_df['expert_grade'].values
    
    # Compute metrics
    class_names = [f"LOCS {i}" for i in range(7)]
    metrics = compute_clinical_metrics(model_predictions, expert_grades, class_names)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    plot_agreement_chart(
        model_predictions,
        expert_grades,
        save_path=output_dir / 'agreement_chart.png'
    )
    
    report = generate_clinical_report(
        metrics,
        output_path=output_dir / 'clinical_report.txt'
    )
    
    print(report)
    
    # Save metrics as JSON
    import json
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in metrics.items()
                    if k != 'classification_report'}
    metrics_json['classification_report'] = metrics['classification_report']
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
