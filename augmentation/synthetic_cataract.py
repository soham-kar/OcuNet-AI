"""
Synthetic Cataract Augmentation for Fundus Images
==================================================

Physically-grounded cataract simulation using classical image processing.
Each degradation step is justified by the underlying optics and clinical
presentation of cataract in fundus photography.

Physical Model
--------------
Cataract is the opacification of the crystalline lens. In fundus imaging,
light must pass through the lens twice (illumination → retina → camera),
so lens opacity causes:

1. Light Scattering → Gaussian blur (Mie/Rayleigh scattering in lens)
2. Reduced Transmission → Contrast loss (fewer photons reach retina)
3. Forward Scatter → Veiling glare / haze (stray light in optical path)
4. Nuclear Sclerosis → Yellow/brown color shift (lens protein changes)
5. Low-light Degradation → Sensor noise (reduced signal-to-noise ratio)
6. Fine Detail Loss → Vessel attenuation (high-frequency suppression)

References
----------
- Kim et al. (2024), "Fundus image enhancement through direct diffusion bridges",
  IEEE JBHI. arXiv:2409.12377 — Clinically validated degradation model
- Gong et al. (2025), "Versatile Cataract Fundus Image Restoration Model",
  Scientific Reports. arXiv:2411.12278 — GAN-based cataract synthesis
- Chylack et al. (1993), "LOCS III: Lens Opacities Classification System",
  Archives of Ophthalmology — Standardized cataract grading
- Peli (1990), "Contrast in complex images", JOSA A — Contrast perception
- Westheimer (2008), "Directional sensitivity of the retina", J. Physiol.

Usage
-----
    from augmentation.synthetic_cataract import SyntheticCataract

    aug = SyntheticCataract(severity='random')
    synthetic_img = aug(clear_fundus_image)

    # Batch generation:
    aug = SyntheticCataract(severity='moderate')
    for img in clear_images:
        cataract_img = aug(img)
"""

import numpy as np
import cv2
from pathlib import Path
import random
from typing import Tuple, Optional, Union, Literal

# ──────────────────────────────────────────────────────────────
# Severity Parameters
# ──────────────────────────────────────────────────────────────
# Each severity level maps to clinically meaningful cataract grades:
#   Mild   ≈ LOCS III NO2-NO3 / C1-C2 (early cataract)
#   Moderate ≈ LOCS III NO4-NO5 / C3-C4 (clinically significant)
#   Severe ≈ LOCS III NO6 / C5 (advanced, pre-surgical)

SEVERITY_PARAMS = {
    'mild': {
        'blur_sigma': (1.5, 2.5),        # σ of Gaussian kernel
        'contrast_factor': (0.75, 0.90),  # 1.0 = no change, 0.0 = full reduction
        'haze_alpha': (0.05, 0.15),       # α for white overlay blending
        'color_shift_strength': (0.05, 0.12),  # yellow/brown tint intensity
        'noise_sigma': (0.005, 0.012),    # Gaussian noise σ (normalized)
        'vessel_attenuation': (0.80, 0.95),  # high-frequency preservation factor
    },
    'moderate': {
        'blur_sigma': (3.0, 4.5),
        'contrast_factor': (0.50, 0.70),
        'haze_alpha': (0.15, 0.30),
        'color_shift_strength': (0.12, 0.22),
        'noise_sigma': (0.012, 0.025),
        'vessel_attenuation': (0.55, 0.75),
    },
    'severe': {
        'blur_sigma': (5.0, 7.5),
        'contrast_factor': (0.30, 0.50),
        'haze_alpha': (0.30, 0.50),
        'color_shift_strength': (0.22, 0.35),
        'noise_sigma': (0.025, 0.045),
        'vessel_attenuation': (0.30, 0.50),
    },
}


class SyntheticCataract:
    """
    Physically-grounded synthetic cataract augmentation for fundus images.

    Simulates the optical effects of a cataractous lens on fundus photography
    through a sequence of interpretable, parameterized degradations.

    Parameters
    ----------
    severity : str, default='random'
        Cataract severity level:
        - 'mild': Early cataract (LOCS III NO2-NO3)
        - 'moderate': Clinically significant (LOCS III NO4-NO5)
        - 'severe': Advanced/pre-surgical (LOCS III NO6)
        - 'random': Randomly selects a severity level per call

    apply_prob : float, default=1.0
        Probability of applying the augmentation (1.0 = always apply).
        Useful for on-the-fly training augmentation.

    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    severity_params : dict
        Current severity parameters (blur_sigma, contrast_factor, etc.)

    Examples
    --------
    >>> aug = SyntheticCataract(severity='moderate')
    >>> img = cv2.imread('clear_fundus.jpg')
    >>> img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    >>> cataract_img = aug(img)
    """

    def __init__(
        self,
        severity: Literal['mild', 'moderate', 'severe', 'random'] = 'random',
        apply_prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.severity = severity
        self.apply_prob = apply_prob

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _sample_params(self) -> dict:
        """Sample severity parameters with randomization within ranges."""
        if self.severity == 'random':
            severity = random.choice(['mild', 'moderate', 'severe'])
        else:
            severity = self.severity

        ranges = SEVERITY_PARAMS[severity]
        params = {}
        for key, (lo, hi) in ranges.items():
            params[key] = random.uniform(lo, hi)
        params['severity_label'] = severity
        return params

    # ──────────────────────────────────────────────────────────
    # Step 1: Gaussian Blur — Light Scattering
    # ──────────────────────────────────────────────────────────
    # Physical basis: The cataractous lens contains protein aggregates
    # that scatter light via Mie and Rayleigh scattering. This acts as
    # a low-pass filter on the retinal image, equivalent to convolving
    # with a Gaussian point-spread function (PSF).
    #
    # σ ∝ cataract density (higher σ = more scattering = more opacity)
    #
    # Reference: Kim et al. (2024), FD3 — "light transmission disturbance"
    #            Westheimer (2008) — optical PSF of the human eye

    def _apply_gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur simulating light scattering through cloudy lens.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8
        sigma : float
            Standard deviation of Gaussian kernel.
            Mild: 1.5-2.5 | Moderate: 3.0-4.5 | Severe: 5.0-7.5

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        # Use kernel size proportional to sigma (cover 99.7% of Gaussian)
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1  # Must be odd

        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # ──────────────────────────────────────────────────────────
    # Step 2: Contrast Reduction — Reduced Light Transmission
    # ──────────────────────────────────────────────────────────
    # Physical basis: The opaque lens reduces the number of photons
    # reaching the retina. This compresses the dynamic range of the
    # fundus image, reducing contrast between bright (optic disc) and
    # dark (vessels, macula) regions.
    #
    # Contrast_factor < 1.0 compresses pixel values toward the mean.
    #
    # Reference: Peli (1990) — contrast definition in complex images
    #            Chylack et al. (1993) — LOCS III grading

    def _reduce_contrast(
        self, image: np.ndarray, factor: float
    ) -> np.ndarray:
        """
        Reduce image contrast simulating reduced retinal illumination.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8
        factor : float
            Contrast multiplier. 1.0 = no change, 0.0 = full reduction.
            Mild: 0.75-0.90 | Moderate: 0.50-0.70 | Severe: 0.30-0.50

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        # Compress toward mean: new = mean + factor * (old - mean)
        reduced = mean + factor * (image.astype(np.float32) - mean)
        return np.clip(reduced, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # Step 3: Haze Overlay — Forward Light Scattering (Veiling Glare)
    # ──────────────────────────────────────────────────────────
    # Physical basis: Forward-scattered light from the opaque lens
    # creates a veiling glare — a whitish veil over the entire image.
    # This is distinct from blur: blur reduces resolution, while haze
    # adds a uniform luminance offset that washes out dark features.
    #
    # Modeled as alpha blending with a white overlay.
    #
    # Reference: Kim et al. (2024), FD3 — "haze" degradation component
    #            Vos (1984) — disability glare and stray light

    def _add_haze(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """
        Add veiling glare (haze) simulating forward light scatter.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8
        alpha : float
            Haze intensity. 0.0 = no haze, 1.0 = full white.
            Mild: 0.05-0.15 | Moderate: 0.15-0.30 | Severe: 0.30-0.50

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        white_overlay = np.ones_like(image, dtype=np.float32) * 255
        hazy = (1 - alpha) * image.astype(np.float32) + alpha * white_overlay
        return np.clip(hazy, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # Step 4: Color Shift — Nuclear Sclerosis (Yellow/Brown Tint)
    # ──────────────────────────────────────────────────────────
    # Physical basis: Nuclear sclerotic cataract causes the lens
    # proteins to denature and accumulate urochrome pigments,
    # producing a characteristic yellow → brown → brunescent color.
    # This selectively absorbs blue light (short wavelengths) while
    # transmitting red/yellow (long wavelengths).
    #
    # Modeled as: reduce blue channel, slightly reduce green,
    # preserve red → net yellow/brown shift.
    #
    # Reference: Chylack et al. (1993), LOCS III — nuclear color grading
    #            Lerman & Borkman (1978) — lens fluorescence & pigmentation

    def _shift_color(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply yellow/brown color shift simulating nuclear sclerosis.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8, RGB order
        strength : float
            Color shift intensity. 0.0 = no shift, 1.0 = extreme.
            Mild: 0.05-0.12 | Moderate: 0.12-0.22 | Severe: 0.22-0.35

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        img = image.astype(np.float32)

        # Blue channel: most absorbed by nuclear sclerosis
        img[:, :, 2] *= (1.0 - strength * 0.6)  # Reduce blue

        # Green channel: moderately absorbed
        img[:, :, 1] *= (1.0 - strength * 0.2)  # Slightly reduce green

        # Red channel: least absorbed (preserved)
        # img[:, :, 0] unchanged

        # Add warm tint (yellow = red + green, brown = red + reduced green)
        warm_tint = np.zeros_like(img)
        warm_tint[:, :, 0] = strength * 30   # Add red
        warm_tint[:, :, 1] = strength * 15   # Add some green (yellow component)

        img = img + warm_tint
        return np.clip(img, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # Step 5: Gaussian Noise — Low-Light Sensor Degradation
    # ──────────────────────────────────────────────────────────
    # Physical basis: Reduced light transmission through the
    # cataractous lens means the camera sensor operates at lower
    # signal-to-noise ratio (SNR). This manifests as increased
    # Gaussian-distributed sensor noise.
    #
    # σ_noise ∝ 1/√(light_level) (photon shot noise)
    #
    # Reference: Healey & Kondepudy (1994) — CCD camera noise models

    def _add_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Add Gaussian noise simulating low-light sensor degradation.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8
        sigma : float
            Noise standard deviation (normalized to [0,1] range).
            Mild: 0.005-0.012 | Moderate: 0.012-0.025 | Severe: 0.025-0.045

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma * 255
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # Step 6: Vessel Attenuation — High-Frequency Suppression
    # ──────────────────────────────────────────────────────────
    # Physical basis: Fine retinal vessels are high-frequency spatial
    # features. Light scattering in the cataractous lens acts as a
    # low-pass filter, preferentially attenuating these fine details
    # while preserving larger structures (optic disc, major vessels).
    #
    # Implemented as: extract high-frequency component via difference
    # of Gaussian, then attenuate it.
    #
    # Reference: Kim et al. (2024), FD3 — "blur" degradation

    def _attenuate_vessels(
        self, image: np.ndarray, factor: float
    ) -> np.ndarray:
        """
        Attenuate fine vessel details via high-frequency suppression.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8
        factor : float
            High-frequency preservation factor. 1.0 = no attenuation,
            0.0 = complete removal of fine details.
            Mild: 0.80-0.95 | Moderate: 0.55-0.75 | Severe: 0.30-0.50

        Returns
        -------
        np.ndarray (H, W, 3), uint8
        """
        # Extract high frequencies via Difference of Gaussian
        sigma_lo = 3.0  # Low-pass (large structures)
        sigma_hi = 0.8  # Band-pass (fine vessels)

        ksize_lo = int(6 * sigma_lo + 1)
        if ksize_lo % 2 == 0:
            ksize_lo += 1
        ksize_hi = int(6 * sigma_hi + 1)
        if ksize_hi % 2 == 0:
            ksize_hi += 1

        blurred_lo = cv2.GaussianBlur(image, (ksize_lo, ksize_lo), sigma_lo)
        blurred_hi = cv2.GaussianBlur(image, (ksize_hi, ksize_hi), sigma_hi)

        # High-frequency = difference between fine and coarse blur
        high_freq = blurred_hi.astype(np.float32) - blurred_lo.astype(np.float32)

        # Attenuate high frequencies
        result = image.astype(np.float32) - (1 - factor) * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────────────────────

    def __call__(
        self,
        image: np.ndarray,
        force_apply: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply full cataract simulation pipeline.

        Parameters
        ----------
        image : np.ndarray (H, W, 3), uint8, RGB
            Clear fundus image.

        force_apply : bool, default=False
            If True, apply regardless of apply_prob.

        Returns
        -------
        degraded_image : np.ndarray (H, W, 3), uint8
            Synthetic cataract fundus image.

        params : dict
            Parameters used for this degradation (for reproducibility).
            Keys: severity_label, blur_sigma, contrast_factor, haze_alpha,
                  color_shift_strength, noise_sigma, vessel_attenuation
        """
        # Probability check
        if not force_apply and random.random() > self.apply_prob:
            return image, {'severity_label': 'none'}

        # Sample parameters
        params = self._sample_params()

        # Ensure RGB uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        result = image.copy()

        # Step 1: Light scattering → Gaussian blur
        result = self._apply_gaussian_blur(result, params['blur_sigma'])

        # Step 2: Reduced transmission → Contrast loss
        result = self._reduce_contrast(result, params['contrast_factor'])

        # Step 3: Forward scatter → Veiling glare / haze
        result = self._add_haze(result, params['haze_alpha'])

        # Step 4: Nuclear sclerosis → Yellow/brown color shift
        result = self._shift_color(result, params['color_shift_strength'])

        # Step 5: Low-light → Sensor noise
        result = self._add_noise(result, params['noise_sigma'])

        # Step 6: Fine detail loss → Vessel attenuation
        result = self._attenuate_vessels(result, params['vessel_attenuation'])

        return result, params

    def generate_batch(
        self,
        images: list,
        severities: Optional[list] = None,
    ) -> list:
        """
        Generate synthetic cataract images for a batch.

        Parameters
        ----------
        images : list of np.ndarray
            List of clear fundus images.

        severities : list of str, optional
            Severity for each image. If None, uses self.severity.

        Returns
        -------
        list of tuple (np.ndarray, dict)
            Each tuple: (degraded_image, params_dict)
        """
        results = []
        for i, img in enumerate(images):
            if severities is not None:
                original_severity = self.severity
                self.severity = severities[i]
                degraded, params = self(force_apply=True)
                self.severity = original_severity
            else:
                degraded, params = self(force_apply=True)
            results.append((degraded, params))
        return results


# ──────────────────────────────────────────────────────────────
# Utility: Generate comparison grid
# ──────────────────────────────────────────────────────────────

def create_comparison_grid(
    original: np.ndarray,
    mild: np.ndarray,
    moderate: np.ndarray,
    severe: np.ndarray,
    labels: bool = True,
) -> np.ndarray:
    """
    Create a 2×2 comparison grid: Original | Mild
                                    Moderate | Severe

    Parameters
    ----------
    original : np.ndarray (H, W, 3), uint8
    mild, moderate, severe : np.ndarray (H, W, 3), uint8
    labels : bool
        Whether to add severity labels.

    Returns
    -------
    np.ndarray (2*H, 2*W, 3), uint8
    """
    h, w = original.shape[:2]

    # Resize all to same size
    def resize(img):
        if img.shape[:2] != (h, w):
            return cv2.resize(img, (w, h))
        return img

    mild_r = resize(mild)
    mod_r = resize(moderate)
    sev_r = resize(severe)

    # Create grid
    top = np.hstack([original, mild_r])
    bottom = np.hstack([mod_r, sev_r])
    grid = np.vstack([top, bottom])

    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = h / 500
        thickness = max(1, int(font_scale * 2))
        color = (255, 255, 255)

        positions = [
            (10, 30, "Original (Clear)"),
            (w + 10, 30, "Mild Cataract"),
            (10, h + 30, "Moderate Cataract"),
            (w + 10, h + 30, "Severe Cataract"),
        ]
        for x, y, text in positions:
            # Add dark background for readability
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(grid, (x, y - th - 5), (x + tw + 10, y + 5),
                          (0, 0, 0), -1)
            cv2.putText(grid, text, (x + 5, y), font, font_scale,
                        color, thickness, cv2.LINE_AA)

    return grid


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    print("=" * 60)
    print("  Synthetic Cataract — Phase 1 Demo")
    print("  Physically-grounded cataract simulation")
    print("=" * 60)

    # Find a source image
    source_dirs = [
        "data/raw/PALM/Training/Classification",
        "data/raw/DDR/DR_grading/test",
        "data/raw/RFMiD/Test_set",
    ]

    source_img = None
    for d in source_dirs:
        if os.path.isdir(d):
            imgs = [f for f in os.listdir(d) if f.endswith(('.jpg', '.png'))]
            if imgs:
                source_img = os.path.join(d, imgs[0])
                break

    if source_img is None:
        print("No source image found. Using synthetic test pattern.")
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        cv2.circle(img, (192, 192), 150, (200, 150, 100), -1)
        cv2.circle(img, (192, 192), 50, (255, 200, 150), -1)
        for i in range(5):
            pt1 = (random.randint(50, 334), random.randint(50, 334))
            pt2 = (random.randint(50, 334), random.randint(50, 334))
            cv2.line(img, pt1, pt2, (50, 20, 20), random.randint(1, 3))
    else:
        print(f"Source: {source_img}")
        img = cv2.imread(source_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate all severities
    aug = SyntheticCataract(severity='random', seed=42)

    mild_img, _ = SyntheticCataract(severity='mild', seed=1)(img, force_apply=True)
    mod_img, p = SyntheticCataract(severity='moderate', seed=2)(img, force_apply=True)
    sev_img, _ = SyntheticCataract(severity='severe', seed=3)(img, force_apply=True)

    print(f"\nModerate params: blur_sigma={p['blur_sigma']:.2f}, "
          f"contrast={p['contrast_factor']:.2f}, haze={p['haze_alpha']:.2f}, "
          f"color_shift={p['color_shift_strength']:.2f}")

    # Create comparison grid
    grid = create_comparison_grid(img, mild_img, mod_img, sev_img)

    # Save
    output_dir = Path("data/synthetic/cataract/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_comparison.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Comparison grid saved to: {output_path}")
    print("=" * 60)
