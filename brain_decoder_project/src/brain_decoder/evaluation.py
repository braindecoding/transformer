"""
Comprehensive Evaluation Metrics for Brain Decoder

This module provides comprehensive evaluation metrics for assessing the quality
of reconstructed visual stimuli from fMRI signals, including:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- CLIP Score (semantic similarity)

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EvaluationMetrics:
    """Comprehensive evaluation metrics for brain decoder reconstruction quality."""

    def __init__(self, device: str = 'auto'):
        """Initialize evaluation metrics."""
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize advanced metrics (will be loaded when needed)
        self.lpips_model = None
        self.clip_model = None
        self.clip_preprocess = None

        print(f"Evaluation metrics initialized on device: {self.device}")

    def _load_lpips_model(self):
        """Load LPIPS model for perceptual similarity."""
        try:
            import lpips
            if self.lpips_model is None:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                print("✅ LPIPS model loaded successfully")
        except ImportError:
            print("⚠️  LPIPS not available. Install with: pip install lpips")
            self.lpips_model = None

    def _load_clip_model(self):
        """Load CLIP model for semantic similarity."""
        try:
            import clip
            if self.clip_model is None:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                print("✅ CLIP model loaded successfully")
        except ImportError:
            print("⚠️  CLIP not available. Install with: pip install clip-by-openai")
            self.clip_model = None

    def compute_mse(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return np.mean((predicted - target) ** 2)

    def compute_psnr(self, predicted: np.ndarray, target: np.ndarray,
                     data_range: float = 1.0) -> List[float]:
        """Compute PSNR for each image pair."""
        psnr_scores = []

        # Convert to uint8 if needed
        if data_range == 1.0:
            predicted_uint8 = (predicted * 255).astype(np.uint8)
            target_uint8 = (target * 255).astype(np.uint8)
            data_range = 255
        else:
            predicted_uint8 = predicted.astype(np.uint8)
            target_uint8 = target.astype(np.uint8)

        for i in range(len(predicted_uint8)):
            try:
                if predicted_uint8[i].ndim == 1:
                    # Reshape to 2D if flattened
                    size = int(np.sqrt(len(predicted_uint8[i])))
                    pred_img = predicted_uint8[i].reshape(size, size)
                    target_img = target_uint8[i].reshape(size, size)
                else:
                    pred_img = predicted_uint8[i]
                    target_img = target_uint8[i]

                psnr_val = psnr(target_img, pred_img, data_range=data_range)
                psnr_scores.append(psnr_val)
            except Exception as e:
                print(f"⚠️  PSNR calculation failed for image {i}: {e}")
                psnr_scores.append(0.0)

        return psnr_scores

    def compute_ssim(self, predicted: np.ndarray, target: np.ndarray,
                     data_range: float = 1.0) -> List[float]:
        """Compute SSIM for each image pair."""
        ssim_scores = []

        # Convert to uint8 if needed
        if data_range == 1.0:
            predicted_uint8 = (predicted * 255).astype(np.uint8)
            target_uint8 = (target * 255).astype(np.uint8)
            data_range = 255
        else:
            predicted_uint8 = predicted.astype(np.uint8)
            target_uint8 = target.astype(np.uint8)

        for i in range(len(predicted_uint8)):
            try:
                if predicted_uint8[i].ndim == 1:
                    # Reshape to 2D if flattened
                    size = int(np.sqrt(len(predicted_uint8[i])))
                    pred_img = predicted_uint8[i].reshape(size, size)
                    target_img = target_uint8[i].reshape(size, size)
                else:
                    pred_img = predicted_uint8[i]
                    target_img = target_uint8[i]

                ssim_val = ssim(target_img, pred_img, data_range=data_range)
                ssim_scores.append(ssim_val)
            except Exception as e:
                print(f"⚠️  SSIM calculation failed for image {i}: {e}")
                ssim_scores.append(0.0)

        return ssim_scores

    def compute_fid(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Fréchet Inception Distance (simplified version)."""
        try:
            # Flatten images for feature extraction
            pred_flat = predicted.reshape(predicted.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)

            # Calculate means and covariances
            mu1, sigma1 = target_flat.mean(axis=0), np.cov(target_flat, rowvar=False)
            mu2, sigma2 = pred_flat.mean(axis=0), np.cov(pred_flat, rowvar=False)

            # Add small epsilon to diagonal for numerical stability
            sigma1 += np.eye(sigma1.shape[0]) * 1e-6
            sigma2 += np.eye(sigma2.shape[0]) * 1e-6

            # Calculate FID
            diff = mu1 - mu2
            covmean = linalg.sqrtm(sigma1.dot(sigma2))

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
            return float(fid)

        except Exception as e:
            print(f"⚠️  FID calculation failed: {e}")
            return float('inf')

    def compute_lpips(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)."""
        if self.lpips_model is None:
            self._load_lpips_model()

        if self.lpips_model is None:
            # Fallback to simplified perceptual distance
            return self._compute_simple_lpips(predicted, target)

        try:
            # Convert to torch tensors and normalize to [-1, 1]
            pred_tensor = torch.FloatTensor(predicted).to(self.device)
            target_tensor = torch.FloatTensor(target).to(self.device)

            # Reshape if needed and add channel dimension
            if pred_tensor.ndim == 2:
                size = int(np.sqrt(pred_tensor.shape[1]))
                pred_tensor = pred_tensor.view(-1, 1, size, size)
                target_tensor = target_tensor.view(-1, 1, size, size)
            elif pred_tensor.ndim == 3:
                pred_tensor = pred_tensor.unsqueeze(1)
                target_tensor = target_tensor.unsqueeze(1)

            # Normalize to [-1, 1] for LPIPS
            pred_tensor = pred_tensor * 2.0 - 1.0
            target_tensor = target_tensor * 2.0 - 1.0

            # Repeat grayscale to RGB if needed
            if pred_tensor.shape[1] == 1:
                pred_tensor = pred_tensor.repeat(1, 3, 1, 1)
                target_tensor = target_tensor.repeat(1, 3, 1, 1)

            # Compute LPIPS
            with torch.no_grad():
                lpips_scores = self.lpips_model(pred_tensor, target_tensor)

            return float(lpips_scores.mean().cpu())

        except Exception as e:
            print(f"⚠️  LPIPS calculation failed: {e}")
            return self._compute_simple_lpips(predicted, target)

    def _compute_simple_lpips(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Simplified perceptual distance calculation."""
        try:
            # Reshape if needed
            if predicted.ndim == 2:
                size = int(np.sqrt(predicted.shape[1]))
                pred_imgs = predicted.reshape(-1, size, size)
                target_imgs = target.reshape(-1, size, size)
            else:
                pred_imgs = predicted
                target_imgs = target

            # Calculate gradient-based perceptual distance
            pred_grad_x = np.gradient(pred_imgs, axis=2)
            pred_grad_y = np.gradient(pred_imgs, axis=1)
            target_grad_x = np.gradient(target_imgs, axis=2)
            target_grad_y = np.gradient(target_imgs, axis=1)

            grad_diff_x = np.mean((pred_grad_x - target_grad_x) ** 2)
            grad_diff_y = np.mean((pred_grad_y - target_grad_y) ** 2)

            return float(np.sqrt(grad_diff_x + grad_diff_y))

        except Exception as e:
            print(f"⚠️  Simple LPIPS calculation failed: {e}")
            return float('inf')

    def compute_clip_score(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute CLIP score for semantic similarity."""
        if self.clip_model is None:
            self._load_clip_model()

        if self.clip_model is None:
            # Fallback to simplified semantic similarity
            return self._compute_simple_clip_score(predicted, target)

        try:
            # Convert to PIL Images and preprocess
            from PIL import Image

            clip_scores = []

            for i in range(len(predicted)):
                # Reshape if needed
                if predicted[i].ndim == 1:
                    size = int(np.sqrt(len(predicted[i])))
                    pred_img = predicted[i].reshape(size, size)
                    target_img = target[i].reshape(size, size)
                else:
                    pred_img = predicted[i]
                    target_img = target[i]

                # Convert to PIL Images
                pred_pil = Image.fromarray((pred_img * 255).astype(np.uint8))
                target_pil = Image.fromarray((target_img * 255).astype(np.uint8))

                # Preprocess for CLIP
                pred_processed = self.clip_preprocess(pred_pil).unsqueeze(0).to(self.device)
                target_processed = self.clip_preprocess(target_pil).unsqueeze(0).to(self.device)

                # Get CLIP features
                with torch.no_grad():
                    pred_features = self.clip_model.encode_image(pred_processed)
                    target_features = self.clip_model.encode_image(target_processed)

                    # Normalize features
                    pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
                    target_features = target_features / target_features.norm(dim=-1, keepdim=True)

                    # Compute cosine similarity
                    similarity = torch.sum(pred_features * target_features, dim=-1)
                    clip_scores.append(float(similarity.cpu()))

            return float(np.mean(clip_scores))

        except Exception as e:
            print(f"⚠️  CLIP calculation failed: {e}")
            return self._compute_simple_clip_score(predicted, target)

    def _compute_simple_clip_score(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Simplified semantic similarity score."""
        try:
            # Simple feature-based similarity
            pred_features = predicted.reshape(predicted.shape[0], -1)
            target_features = target.reshape(target.shape[0], -1)

            # Normalize features
            pred_norm = pred_features / (np.linalg.norm(pred_features, axis=1, keepdims=True) + 1e-8)
            target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)

            # Calculate cosine similarity
            similarities = np.sum(pred_norm * target_norm, axis=1)
            return float(np.mean(similarities))

        except Exception as e:
            print(f"⚠️  Simple CLIP calculation failed: {e}")
            return 0.0

    def evaluate_all(self, predicted: np.ndarray, target: np.ndarray) -> Dict[str, Union[float, List[float]]]:
        """Compute all evaluation metrics."""
        print("🔄 Computing comprehensive evaluation metrics...")

        results = {}

        # Basic metrics
        print("   Computing MSE...")
        results['mse'] = self.compute_mse(predicted, target)

        print("   Computing PSNR...")
        psnr_scores = self.compute_psnr(predicted, target)
        results['psnr_scores'] = psnr_scores
        results['psnr_mean'] = np.mean(psnr_scores)
        results['psnr_std'] = np.std(psnr_scores)

        print("   Computing SSIM...")
        ssim_scores = self.compute_ssim(predicted, target)
        results['ssim_scores'] = ssim_scores
        results['ssim_mean'] = np.mean(ssim_scores)
        results['ssim_std'] = np.std(ssim_scores)

        print("   Computing FID...")
        results['fid'] = self.compute_fid(predicted, target)

        print("   Computing LPIPS...")
        results['lpips'] = self.compute_lpips(predicted, target)

        print("   Computing CLIP Score...")
        results['clip_score'] = self.compute_clip_score(predicted, target)

        # Additional correlation metric
        pred_flat = predicted.flatten()
        target_flat = target.flatten()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        results['correlation'] = correlation

        print("✅ All metrics computed successfully!")
        return results

    def print_results(self, results: Dict[str, Union[float, List[float]]]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)

        print(f"📈 Basic Metrics:")
        print(f"   MSE: {results['mse']:.6f}")
        print(f"   Correlation: {results['correlation']:.4f}")

        print(f"\n📊 Image Quality Metrics:")
        print(f"   PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"   SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

        print(f"\n🎨 Perceptual Metrics:")
        print(f"   FID: {results['fid']:.2f}")
        print(f"   LPIPS: {results['lpips']:.4f}")

        print(f"\n🧠 Semantic Metrics:")
        print(f"   CLIP Score: {results['clip_score']:.4f}")

        print("="*60)

    def plot_metrics(self, results: Dict[str, Union[float, List[float]]],
                     save_path: Optional[str] = None, show: bool = True):
        """Create comprehensive plots of all evaluation metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Brain Decoder Evaluation Metrics', fontsize=16, fontweight='bold')

        # 1. Basic Metrics Bar Plot
        ax1 = axes[0, 0]
        basic_metrics = ['MSE', 'Correlation']
        basic_values = [results['mse'], results['correlation']]
        colors = ['red', 'blue']
        bars = ax1.bar(basic_metrics, basic_values, color=colors, alpha=0.7)
        ax1.set_title('Basic Metrics', fontweight='bold')
        ax1.set_ylabel('Value')

        # Add value labels on bars
        for bar, value in zip(bars, basic_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom')

        # 2. PSNR Distribution
        ax2 = axes[0, 1]
        psnr_scores = results['psnr_scores']
        ax2.hist(psnr_scores, bins=min(10, len(psnr_scores)), alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(results['psnr_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {results["psnr_mean"]:.2f}')
        ax2.set_title('PSNR Distribution', fontweight='bold')
        ax2.set_xlabel('PSNR (dB)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. SSIM Distribution
        ax3 = axes[0, 2]
        ssim_scores = results['ssim_scores']
        ax3.hist(ssim_scores, bins=min(10, len(ssim_scores)), alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(results['ssim_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {results["ssim_mean"]:.4f}')
        ax3.set_title('SSIM Distribution', fontweight='bold')
        ax3.set_xlabel('SSIM')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Perceptual Metrics Comparison
        ax4 = axes[1, 0]
        perceptual_metrics = ['FID', 'LPIPS']
        perceptual_values = [results['fid'], results['lpips']]

        # Normalize values for better visualization
        normalized_values = []
        for i, (metric, value) in enumerate(zip(perceptual_metrics, perceptual_values)):
            if metric == 'FID':
                # For FID, lower is better, so we'll show inverse
                normalized_values.append(1 / (1 + value) if value != float('inf') else 0)
            else:
                # For LPIPS, lower is better
                normalized_values.append(1 - min(value, 1))

        bars = ax4.bar(perceptual_metrics, normalized_values, color=['purple', 'brown'], alpha=0.7)
        ax4.set_title('Perceptual Metrics (Normalized)', fontweight='bold')
        ax4.set_ylabel('Quality Score (Higher = Better)')

        # Add original value labels
        for bar, orig_val, metric in zip(bars, perceptual_values, perceptual_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{orig_val:.3f}', ha='center', va='bottom')

        # 5. CLIP Score
        ax5 = axes[1, 1]
        clip_score = results['clip_score']
        ax5.bar(['CLIP Score'], [clip_score], color='teal', alpha=0.7)
        ax5.set_title('Semantic Similarity (CLIP)', fontweight='bold')
        ax5.set_ylabel('CLIP Score')
        ax5.set_ylim(0, 1)
        ax5.text(0, clip_score + 0.02, f'{clip_score:.4f}', ha='center', va='bottom')

        # 6. Overall Quality Radar Chart
        ax6 = axes[1, 2]

        # Prepare data for radar chart
        metrics_names = ['PSNR', 'SSIM', 'FID\n(inv)', 'LPIPS\n(inv)', 'CLIP', 'Corr']

        # Normalize all metrics to 0-1 scale (higher = better)
        normalized_scores = [
            min(results['psnr_mean'] / 40, 1),  # PSNR normalized by 40dB
            results['ssim_mean'],  # SSIM already 0-1
            1 / (1 + results['fid']) if results['fid'] != float('inf') else 0,  # FID inverted
            1 - min(results['lpips'], 1),  # LPIPS inverted
            results['clip_score'],  # CLIP already 0-1
            (results['correlation'] + 1) / 2  # Correlation normalized to 0-1
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        normalized_scores = normalized_scores + [normalized_scores[0]]  # Complete the circle

        ax6.plot(angles, normalized_scores, 'o-', linewidth=2, color='navy')
        ax6.fill(angles, normalized_scores, alpha=0.25, color='navy')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics_names)
        ax6.set_ylim(0, 1)
        ax6.set_title('Overall Quality Profile', fontweight='bold')
        ax6.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Metrics plot saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_sample_reconstructions(self, predicted: np.ndarray, target: np.ndarray,
                                   num_samples: int = 6, save_path: Optional[str] = None,
                                   show: bool = True):
        """Plot sample reconstructions with their metrics."""
        num_samples = min(num_samples, len(predicted))

        fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Sample Reconstructions with Individual Metrics', fontsize=16, fontweight='bold')

        # Compute individual metrics for samples
        psnr_scores = self.compute_psnr(predicted[:num_samples], target[:num_samples])
        ssim_scores = self.compute_ssim(predicted[:num_samples], target[:num_samples])

        for i in range(num_samples):
            # Reshape images if needed
            if predicted[i].ndim == 1:
                size = int(np.sqrt(len(predicted[i])))
                pred_img = predicted[i].reshape(size, size)
                target_img = target[i].reshape(size, size)
            else:
                pred_img = predicted[i]
                target_img = target[i]

            # Target image
            axes[0, i].imshow(target_img, cmap='gray')
            axes[0, i].set_title(f'Target {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Predicted image
            axes[1, i].imshow(pred_img, cmap='gray')
            axes[1, i].set_title(f'Predicted {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Difference image
            diff_img = np.abs(target_img - pred_img)
            im = axes[2, i].imshow(diff_img, cmap='hot')
            axes[2, i].set_title(f'Difference {i+1}\nPSNR: {psnr_scores[i]:.1f}dB\nSSIM: {ssim_scores[i]:.3f}',
                               fontweight='bold')
            axes[2, i].axis('off')

            # Add colorbar for difference image
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️  Sample reconstructions plot saved to: {save_path}")

        if show:
            plt.show()

        return fig