"""
Temperature Scaling for GLAAM Multi-Label Calibration

CRITICAL: Your current predictions.pkl contains PROBABILITIES (post-sigmoid).
Temperature scaling needs LOGITS (pre-sigmoid).

This script:
1. Converts probabilities back to logits (inverse sigmoid)
2. Applies per-disease temperature scaling
3. Reduces ECE from 23.46% to target < 15%

Run: python evaluation/temperature_scaling.py
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
import json


def inverse_sigmoid(p, epsilon=1e-7):
    """
    Convert probability back to logit.
    logit = log(p / (1-p))
    
    Note: This is an approximation since we lost information during sigmoid.
    Ideally, save raw logits during inference.
    """
    # Clip probabilities to avoid log(0)
    p_clipped = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p_clipped / (1 - p_clipped))


def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error for binary classification.
    
    Args:
        probs: probabilities (n_samples,)
        labels: binary labels (n_samples,)
        n_bins: number of bins for calibration
    
    Returns:
        ECE as a percentage
    """
    # Ensure binary
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    
    # Confidence = probability of predicted class
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
    
    return ece * 100  # Return as percentage


class BinaryTemperatureScaler:
    """
    Temperature scaling for single binary classification task.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.history = []
    
    def fit(self, logits, labels, n_bins=15, verbose=True):
        """
        Find optimal temperature to minimize ECE.
        
        Args:
            logits: raw logits before sigmoid (n_samples,)
            labels: binary labels (n_samples,)
        """
        logits = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()
        
        def objective(T):
            """ECE after temperature scaling."""
            scaled_logits = logits / T
            scaled_probs = sigmoid(scaled_logits)
            ece = compute_ece(scaled_probs, labels, n_bins=n_bins)
            self.history.append((T, ece))
            return ece
        
        # Optimize temperature
        result = minimize_scalar(
            objective,
            bounds=(0.1, 10.0),
            method='bounded',
            options={'xatol': 1e-5, 'maxiter': 100}
        )
        
        self.temperature = result.x
        
        if verbose:
            ece_before = compute_ece(sigmoid(logits), labels, n_bins=n_bins)
            ece_after = compute_ece(sigmoid(logits / self.temperature), labels, n_bins=n_bins)
            print(f"    T={self.temperature:.3f}, ECE: {ece_before:.2f}% → {ece_after:.2f}% (Δ={ece_before - ece_after:.2f}%)")
        
        return self
    
    def transform(self, logits):
        """Apply temperature scaling to logits."""
        return sigmoid(logits / self.temperature)


class GLAAMCalibrator:
    """
    Multi-label temperature scaling for GLAAM.
    Fits independent temperature for each disease.
    """
    
    def __init__(self, disease_names):
        self.disease_names = disease_names
        self.scalers = {}
        self.results = {}
    
    def fit(self, predictions, labels, n_bins=15):
        """
        Fit temperature scaling for each disease.
        
        Args:
            predictions: shape (n_samples, n_diseases) - PROBABILITIES
            labels: shape (n_samples, n_diseases) - binary labels
        """
        print("=" * 70)
        print("GLAAM TEMPERATURE SCALING CALIBRATION")
        print("=" * 70)
        print("⚠️  WARNING: Converting probabilities → logits (inverse sigmoid)")
        print("    For best results, save raw logits during training.\n")
        
        n_samples, n_diseases = predictions.shape
        
        for i, disease in enumerate(self.disease_names):
            print(f"\n{disease}:")
            
            # Extract disease-specific data
            probs = predictions[:, i]
            disease_labels = labels[:, i]
            
            # Convert probabilities back to logits
            logits = inverse_sigmoid(probs)
            
            # Fit temperature scaler
            scaler = BinaryTemperatureScaler()
            scaler.fit(logits, disease_labels, n_bins=n_bins, verbose=True)
            
            self.scalers[disease] = scaler
            
            # Store results
            ece_before = compute_ece(probs, disease_labels, n_bins=n_bins)
            ece_after = compute_ece(scaler.transform(logits), disease_labels, n_bins=n_bins)
            auc = roc_auc_score(disease_labels, probs)
            
            self.results[disease] = {
                'temperature': scaler.temperature,
                'ece_before': ece_before,
                'ece_after': ece_after,
                'auc': auc,
                'n_positive': int(disease_labels.sum()),
                'n_total': len(disease_labels)
            }
        
        self._print_summary()
        return self
    
    def transform(self, predictions):
        """
        Apply temperature scaling to predictions.
        
        Args:
            predictions: shape (n_samples, n_diseases) - PROBABILITIES
        
        Returns:
            calibrated_predictions: shape (n_samples, n_diseases)
        """
        calibrated = np.zeros_like(predictions)
        
        for i, disease in enumerate(self.disease_names):
            probs = predictions[:, i]
            logits = inverse_sigmoid(probs)
            calibrated[:, i] = self.scalers[disease].transform(logits)
        
        return calibrated
    
    def _print_summary(self):
        """Print calibration summary table."""
        print("\n" + "=" * 70)
        print("CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"{'Disease':<12} {'Temp':<8} {'ECE Before':<12} {'ECE After':<12} {'Δ ECE':<10} {'AUC':<8}")
        print("-" * 70)
        
        ece_before_list = []
        ece_after_list = []
        
        for disease in self.disease_names:
            res = self.results[disease]
            delta = res['ece_before'] - res['ece_after']
            ece_before_list.append(res['ece_before'])
            ece_after_list.append(res['ece_after'])
            
            print(f"{disease:<12} {res['temperature']:<8.3f} "
                  f"{res['ece_before']:<12.2f}% {res['ece_after']:<12.2f}% "
                  f"{delta:>9.2f}% {res['auc']:<8.4f}")
        
        print("-" * 70)
        mean_before = np.mean(ece_before_list)
        mean_after = np.mean(ece_after_list)
        mean_delta = mean_before - mean_after
        
        print(f"{'Mean':<12} {'':<8} {mean_before:<12.2f}% {mean_after:<12.2f}% {mean_delta:>9.2f}%")
        print("=" * 70)
        
        # Assessment
        if mean_after < 5:
            print("✅ EXCELLENT: Mean ECE < 5% (suitable for FDA review)")
        elif mean_after < 10:
            print("✅ GOOD: Mean ECE < 10% (acceptable for research)")
        elif mean_after < 15:
            print("⚠️  FAIR: Mean ECE < 15% (marginal, consider further calibration)")
        else:
            print("❌ POOR: Mean ECE >= 15% (not suitable for clinical use)")
            print("   Recommendation: Try Platt scaling or isotonic regression")
    
    def plot_reliability_diagrams(self, predictions, labels, save_path=None):
        """Generate before/after reliability diagrams."""
        fig, axes = plt.subplots(2, len(self.disease_names), figsize=(20, 10))
        fig.suptitle('GLAAM Calibration: Before vs After Temperature Scaling', 
                     fontsize=16, fontweight='bold')
        
        for i, disease in enumerate(self.disease_names):
            probs = predictions[:, i]
            disease_labels = labels[:, i]
            logits = inverse_sigmoid(probs)
            
            # Before
            ax_before = axes[0, i]
            ece_before = self._plot_single_reliability(
                ax_before, probs, disease_labels, 
                f"{disease}\nBefore (T=1.0)"
            )
            
            # After
            probs_after = self.scalers[disease].transform(logits)
            ax_after = axes[1, i]
            ece_after = self._plot_single_reliability(
                ax_after, probs_after, disease_labels,
                f"{disease}\nAfter (T={self.scalers[disease].temperature:.2f})"
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved reliability diagrams: {save_path}")
        
        plt.show()
    
    def _plot_single_reliability(self, ax, probs, labels, title, n_bins=10):
        """Plot a single reliability diagram."""
        predictions = (probs > 0.5).astype(int)
        confidences = np.where(predictions == 1, probs, 1 - probs)
        accuracies = (predictions == labels).astype(float)
        
        # Binning
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
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
        
        # ECE calculation
        total_samples = len(probs)
        ece = sum([abs(bin_confs[i] - bin_accs[i]) * (bin_counts[i] / total_samples) 
                   for i in range(n_bins) if bin_counts[i] > 0]) * 100
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect', alpha=0.7)
        
        # Plot bars for accuracy
        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.6, 
               color='steelblue', edgecolor='black', label='Accuracy')
        
        # Plot confidence line
        ax.plot(bin_centers, bin_confs, 'ro-', linewidth=2, 
                markersize=8, label='Confidence')
        
        # Highlight gaps
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ax.plot([bin_confs[i], bin_confs[i]], 
                       [bin_accs[i], bin_confs[i]], 
                       'r-', alpha=0.3, linewidth=3)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{title}\nECE: {ece:.2f}%', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        return ece
    
    def save(self, path):
        """Save calibrator to file."""
        data = {
            'disease_names': self.disease_names,
            'temperatures': {d: self.scalers[d].temperature for d in self.disease_names},
            'results': self.results
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved calibrator to: {path}")
    
    @classmethod
    def load(cls, path):
        """Load calibrator from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        calibrator = cls(data['disease_names'])
        for disease, temp in data['temperatures'].items():
            scaler = BinaryTemperatureScaler()
            scaler.temperature = temp
            calibrator.scalers[disease] = scaler
        calibrator.results = data['results']
        
        return calibrator


def main():
    """Main calibration pipeline."""
    print("\n" + "=" * 70)
    print("GLAAM TEMPERATURE SCALING")
    print("=" * 70)
    
    # Load predictions
    predictions_path = Path("checkpoints_glaam/glaam_final_predictions.pkl")
    print(f"\n📂 Loading predictions from: {predictions_path}")
    
    with open(predictions_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = data['predictions']  # (n_samples, 4) - probabilities
    labels = data['labels']            # (n_samples, 4) - binary labels
    disease_names = data['disease_names']  # ['Cataract', 'DR', 'Glaucoma', 'Myopia']
    
    print(f"   Loaded {len(predictions)} samples")
    print(f"   Diseases: {disease_names}\n")
    
    # Split into train/val for calibration
    # Use random 80/20 split (in practice, use your actual val set)
    n_samples = len(predictions)
    n_val = int(0.2 * n_samples)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    
    val_predictions = predictions[val_indices]
    val_labels = labels[val_indices]
    
    print(f"📊 Using {len(val_predictions)} samples for calibration\n")
    
    # Fit temperature scaling
    calibrator = GLAAMCalibrator(disease_names)
    calibrator.fit(val_predictions, val_labels, n_bins=15)
    
    # Generate reliability diagrams
    output_dir = Path("calibration_results")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "glaam_temperature_scaling.png"
    calibrator.plot_reliability_diagrams(val_predictions, val_labels, save_path=plot_path)
    
    # Save calibrator
    calibrator_path = output_dir / "glaam_calibrator.json"
    calibrator.save(calibrator_path)
    
    # Apply to full dataset and save
    calibrated_predictions = calibrator.transform(predictions)
    
    output_data = {
        'predictions': calibrated_predictions,
        'predictions_original': predictions,
        'labels': labels,
        'paths': data['paths'],
        'disease_names': disease_names,
        'calibrator': {d: calibrator.scalers[d].temperature for d in disease_names}
    }
    
    output_path = Path("checkpoints_glaam/glaam_final_predictions_calibrated.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n💾 Saved calibrated predictions: {output_path}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Re-run uncertainty_analysis.py with calibrated predictions")
    print("2. Update training pipeline to save RAW LOGITS (not probabilities)")
    print("3. Apply calibrator during inference:")
    print("   calibrator = GLAAMCalibrator.load('calibration_results/glaam_calibrator.json')")
    print("   calibrated_probs = calibrator.transform(model_predictions)")
    print("=" * 70)


if __name__ == "__main__":
    main()
