"""
Proper Temperature Scaling Using Raw Logits

CRITICAL FIX: Uses raw logits from model, not inverse sigmoid conversion.
Requires: glaam_final_predictions.pkl with 'logits' key

Run: python evaluation/temperature_scaling_proper.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from path lib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error for binary classification.
    
    Args:
        probs: probabilities (n_samples,)
        labels: binary labels (n_samples,)
        n_bins: number of bins
    
    Returns:
        ECE as percentage
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    
    # Binary confidence
    predictions = (probs > 0.5).astype(int)
    confidences = np.where(predictions == 1, probs, 1 - probs)
    accuracies = (predictions == labels).astype(float)
    
    # Binning
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece * 100


class ProperTemperatureScaler(nn.Module):
    """
    Temperature scaling using RAW LOGITS (no inverse sigmoid).
    Each disease gets its own temperature parameter.
    """
    
    def __init__(self, num_diseases=4):
        super().__init__()
        # Initialize T=1.0 (no scaling)
        self.temperatures = nn.Parameter(torch.ones(num_diseases))
    
    def forward(self, logits):
        """
        Args:
            logits: (batch_size, num_diseases) - raw model outputs
        Returns:
            scaled_logits: (batch_size, num_diseases)
        """
        # Clamp temperatures to reasonable range
        T_clamped = self.temperatures.clamp(min=0.1, max=10.0)
        return logits / T_clamped
    
    def fit(self, logits, labels, max_iter=100, lr=0.01, verbose=True):
        """
        Fit temperatures using NLL loss.
        
        Args:
            logits: (n_samples, n_diseases) - numpy or torch tensor
            labels: (n_samples, n_diseases) - binary labels
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float()
        
        optimizer = optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            # Binary cross-entropy with logits (numerically stable)
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, labels
            )
            loss.backward()
            return loss
        
        # Optimize
        optimizer.step(closure)
        
        if verbose:
            final_temps = self.temperatures.detach().cpu().numpy()
            print(f"   Optimized temperatures: {final_temps}")
        
        return self
    
    def predict_proba(self, logits):
        """Get calibrated probabilities."""
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        
        self.eval()
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs = torch.sigmoid(scaled_logits)
        return probs.numpy()


def plot_reliability_diagrams(probs_before, probs_after, labels, disease_names, 
                               temps, save_path):
    """Generate before/after reliability diagrams."""
    fig, axes = plt.subplots(2, len(disease_names), figsize=(20, 10))
    fig.suptitle('GLAAM Temperature Scaling: Raw Logits Method',
                 fontsize=16, fontweight='bold')
    
    for i, disease in enumerate(disease_names):
        # Before
        ax_before = axes[0, i]
        ece_before = _plot_single(ax_before, probs_before[:, i], labels[:, i],
                                   f"{disease}\nBefore (T=1.0)")
        
        # After
        ax_after = axes[1, i]
        ece_after = _plot_single(ax_after, probs_after[:, i], labels[:, i],
                                  f"{disease}\nAfter (T={temps[i]:.2f})")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved reliability diagrams: {save_path}")
    plt.close()


def _plot_single(ax, probs, labels, title, n_bins=10):
    """Plot single reliability diagram."""
    predictions = (probs > 0.5).astype(int)
    confidences = np.where(predictions == 1, probs, 1 - probs)
    accuracies = (predictions == labels).astype(float)
    
    # Binning
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []
    
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = in_bin.sum()
        
        if count > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(count)
        else:
            bin_accs.append(0)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)
    
    # ECE
    total = len(probs)
    ece = sum([abs(bin_confs[i] - bin_accs[i]) * (bin_counts[i] / total)
               for i in range(n_bins) if bin_counts[i] > 0]) * 100
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect')
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.6,
           color='steelblue', edgecolor='black', label='Accuracy')
    ax.plot(bin_centers, bin_confs, 'ro-', linewidth=2, markersize=8, label='Confidence')
    
    # Gaps
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.plot([bin_confs[i], bin_confs[i]], [bin_accs[i], bin_confs[i]],
                   'r-', alpha=0.3, linewidth=3)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{title}\nECE: {ece:.2f}%', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return ece


def main():
    print("=" * 70)
    print("PROPER TEMPERATURE SCALING (USING RAW LOGITS)")
    print("=" * 70)
    
    # Load predictions
    pred_path = Path("checkpoints_glaam/glaam_final_predictions.pkl")
    
    if not pred_path.exists():
        print(f"❌ File not found: {pred_path}")
        print("\n   This script requires predictions with RAW LOGITS.")
        print("   Please re-run training with updated modal_train_glaam_odir.py")
        return
    
    print(f"\n📂 Loading: {pred_path}")
    
    with open(pred_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if logits exist
    if 'logits' not in data:
        print("\n❌ ERROR: No 'logits' key in predictions file!")
        print("   Current keys:", list(data.keys()))
        print("\n   ACTION REQUIRED:")
        print("   1. Re-run GLAAM training with updated modal_train_glaam_odir.py")
        print("   2. This will save raw logits in predictions.pkl")
        print("   3. Then run this script again")
        return
    
    logits = data['logits']  # (n_samples, 4) - raw logits
    probs = data['predictions']  # (n_samples, 4) - probabilities
    labels = data['labels']  # (n_samples, 4) - binary labels
    disease_names = data['disease_names']
    
    n_samples, n_diseases = logits.shape
    
    print(f"   ✅ Loaded {n_samples} samples")
    print(f"   ✅ Found raw logits: shape {logits.shape}")
    print(f"   Diseases: {disease_names}\n")
    
    # Split calibration/validation (proper split, not random)
    # Use first 80% for calibration, last 20% for validation
    split_idx = int(0.8 * n_samples)
    
    cal_logits, val_logits = logits[:split_idx], logits[split_idx:]
    cal_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"📊 Split:")
    print(f"   Calibration: {len(cal_logits)} samples")
    print(f"   Validation: {len(val_logits)} samples\n")
    
    print("=" * 70)
    print("FITTING TEMPERATURE SCALING")
    print("=" * 70)
    
    # Fit scaler
    scaler = ProperTemperatureScaler(num_diseases=n_diseases)
    scaler.fit(cal_logits, cal_labels, max_iter=100, verbose=True)
    
    temps = scaler.temperatures.detach().cpu().numpy()
    
    # Evaluate on VALIDATION set (held-out)
    print("\n" + "=" * 70)
    print("VALIDATION SET RESULTS (HELD-OUT)")
    print("=" * 70)
    
    val_probs_before = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
    val_probs_after = scaler.predict_proba(val_logits)
    
    print(f"\n{'Disease':<12} {'Temp':<8} {'ECE Before':<12} {'ECE After':<12} {'Improvement':<12} {'AUC':<8}")
    print("-" * 70)
    
    ece_before_list, ece_after_list = [], []
    
    for i, disease in enumerate(disease_names):
        ece_before = compute_ece(val_probs_before[:, i], val_labels[:, i])
        ece_after = compute_ece(val_probs_after[:, i], val_labels[:, i])
        auc = roc_auc_score(val_labels[:, i], val_probs_before[:, i])
        
        ece_before_list.append(ece_before)
        ece_after_list.append(ece_after)
        
        improvement = ece_before - ece_after
        
        print(f"{disease:<12} {temps[i]:<8.3f} {ece_before:>10.2f}% {ece_after:>10.2f}% "
              f"{improvement:>11.2f}% {auc:<8.4f}")
    
    print("-" * 70)
    mean_before = np.mean(ece_before_list)
    mean_after = np.mean(ece_after_list)
    mean_improvement = mean_before - mean_after
    
    print(f"{'MEAN':<12} {'':<8} {mean_before:>10.2f}% {mean_after:>10.2f}% "
          f"{mean_improvement:>11.2f}%")
    print("=" * 70)
    
    # Assessment
    print("\n⭐ ASSESSMENT:")
    if mean_after < 5:
        print("   ✅ EXCELLENT: Suitable for FDA clinical validation")
    elif mean_after < 10:
        print("   ✅ GOOD: Suitable for clinical deployment")
    elif mean_after < 15:
        print("   ⚠️  FAIR: Acceptable for research publications")
    else:
        print("   ❌ POOR: Needs additional calibration methods")
    
    # Temperature interpretation
    print("\n🌡️  TEMPERATURE INTERPRETATION:")
    if all(t > 1.0 for t in temps):
        print("   All T > 1.0 → Model was OVERCONFIDENT (expected)")
    elif all(t < 1.0 for t in temps):
        print("   All T < 1.0 → Model was UNDERCONFIDENT (unusual)")
    else:
        print("   Mixed T values → Disease-specific calibration needed")
    
    for i, (disease, t) in enumerate(zip(disease_names, temps)):
        if t > 1.5:
            print(f"   - {disease}: T={t:.2f} (very overconfident)")
        elif t > 1.0:
            print(f"   - {disease}: T={t:.2f} (moderately overconfident)")
        elif t < 0.7:
            print(f"   - {disease}: T={t:.2f} (very underconfident)")
        elif t < 1.0:
            print(f"   - {disease}: T={t:.2f} (moderately underconfident)")
        else:
            print(f"   - {disease}: T={t:.2f} (well-calibrated)")
    
    # Generate plots
    output_dir = Path("calibration_results")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "glaam_calibration_proper.png"
    plot_reliability_diagrams(val_probs_before, val_probs_after, val_labels,
                               disease_names, temps, plot_path)
    
    # Save scaler
    scaler_path = output_dir / "temperature_scaler.pth"
    torch.save({
        'temperatures': temps,
        'disease_names': disease_names,
        'state_dict': scaler.state_dict()
    }, scaler_path)
    print(f"💾 Saved scaler: {scaler_path}")
    
    # Save calibrated predictions for full dataset
    all_probs_calibrated = scaler.predict_proba(logits)
    
    output_data = {
        'predictions': all_probs_calibrated,
        'predictions_uncalibrated': probs,
        'logits': logits,
        'labels': labels,
        'paths': data['paths'],
        'disease_names': disease_names,
        'temperatures': temps
    }
    
    output_path = Path("checkpoints_glaam/glaam_predictions_calibrated.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"💾 Saved calibrated predictions: {output_path}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Use glaam_predictions_calibrated.pkl for all future analysis")
    print("2. Report BOTH uncalibrated and calibrated ECE in paper:")
    print(f"   - Before: {mean_before:.2f}%")
    print(f"   - After: {mean_after:.2f}%")
    print("3. Update XAI report with calibrated metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
