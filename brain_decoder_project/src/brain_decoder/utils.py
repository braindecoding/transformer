"""
Utilities for Brain Decoder

This module provides evaluation metrics, visualization tools, and data
preparation utilities for the brain decoder system.

Author: AI Assistant
Date: 2024
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error
try:
    from sklearn.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
from scipy.stats import pearsonr
import warnings


class BrainDecoderEvaluator:
    """Evaluation metrics for brain decoder performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_mse(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return mean_squared_error(target.flatten(), predicted.flatten())
    
    def compute_psnr(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = self.compute_mse(predicted, target)
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0  # Assuming normalized images
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def compute_ssim(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Structural Similarity Index."""
        if not HAS_SKIMAGE:
            warnings.warn("scikit-image not available, SSIM set to 0")
            return 0.0
            
        if predicted.ndim == 4:  # Batch of images
            ssim_scores = []
            for i in range(predicted.shape[0]):
                for j in range(predicted.shape[1]):  # Time dimension
                    pred_img = predicted[i, j]
                    target_img = target[i, j]
                    
                    if pred_img.ndim == 3 and pred_img.shape[2] == 1:
                        pred_img = pred_img.squeeze(-1)
                        target_img = target_img.squeeze(-1)
                    
                    ssim = structural_similarity(target_img, pred_img, data_range=1.0)
                    ssim_scores.append(ssim)
            
            return np.mean(ssim_scores)
        else:
            return structural_similarity(target, predicted, data_range=1.0)
    
    def compute_correlation(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        pred_flat = predicted.flatten()
        target_flat = target.flatten()
        
        correlation, _ = pearsonr(pred_flat, target_flat)
        return correlation
    
    def compute_pixel_accuracy(self, predicted: np.ndarray, target: np.ndarray, 
                             threshold: float = 0.1) -> float:
        """Compute pixel-wise accuracy within threshold."""
        diff = np.abs(predicted - target)
        accurate_pixels = np.sum(diff < threshold)
        total_pixels = np.prod(predicted.shape)
        
        return accurate_pixels / total_pixels
    
    def evaluate_reconstruction(self, predicted: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of reconstruction quality.
        
        Args:
            predicted: Predicted stimuli
            target: Target stimuli
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = self.compute_mse(predicted, target)
        metrics['psnr'] = self.compute_psnr(predicted, target)
        metrics['correlation'] = self.compute_correlation(predicted, target)
        metrics['pixel_accuracy'] = self.compute_pixel_accuracy(predicted, target)
        
        # SSIM (handle potential errors)
        try:
            metrics['ssim'] = self.compute_ssim(predicted, target)
        except Exception as e:
            warnings.warn(f"SSIM computation failed: {e}")
            metrics['ssim'] = 0.0
        
        # Additional metrics
        metrics['mae'] = np.mean(np.abs(predicted - target))
        metrics['std_error'] = np.std(predicted - target)
        
        return metrics


class BrainDecoderVisualizer:
    """Visualization tools for brain decoder results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE curves
        axes[0, 1].plot(history['train_mse'], label='Train MSE', color='blue')
        if 'val_mse' in history and history['val_mse']:
            axes[0, 1].plot(history['val_mse'], label='Val MSE', color='red')
        axes[0, 1].set_title('Mean Squared Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], color='green')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Loss comparison
        if 'val_loss' in history and history['val_loss']:
            epochs = range(len(history['train_loss']))
            axes[1, 1].plot(epochs, history['train_loss'], label='Train', alpha=0.7)
            axes[1, 1].plot(epochs, history['val_loss'], label='Validation', alpha=0.7)
            axes[1, 1].fill_between(epochs, history['train_loss'], alpha=0.3)
            axes[1, 1].fill_between(epochs, history['val_loss'], alpha=0.3)
            axes[1, 1].set_title('Loss Comparison')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_reconstruction_comparison(self, 
                                     predicted: np.ndarray, 
                                     target: np.ndarray,
                                     num_samples: int = 5,
                                     save_path: Optional[str] = None):
        """Plot comparison between predicted and target stimuli."""
        
        # Select random samples
        if predicted.shape[0] > num_samples:
            indices = np.random.choice(predicted.shape[0], num_samples, replace=False)
            pred_samples = predicted[indices]
            target_samples = target[indices]
        else:
            pred_samples = predicted
            target_samples = target
            num_samples = predicted.shape[0]
        
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Handle different data shapes
            if pred_samples.ndim == 4:  # (batch, time, height, width)
                pred_img = pred_samples[i, 0]  # Take first time step
                target_img = target_samples[i, 0]
            elif pred_samples.ndim == 5:  # (batch, time, channels, height, width)
                pred_img = pred_samples[i, 0, 0]  # Take first time step and channel
                target_img = target_samples[i, 0, 0]
            else:
                pred_img = pred_samples[i]
                target_img = target_samples[i]
            
            # Ensure 2D images
            if pred_img.ndim == 3 and pred_img.shape[0] == 1:
                pred_img = pred_img.squeeze(0)
                target_img = target_img.squeeze(0)
            
            # Target
            axes[0, i].imshow(target_img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Target {i+1}')
            axes[0, i].axis('off')
            
            # Predicted
            axes[1, i].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Predicted {i+1}')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(target_img - pred_img)
            im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'Difference {i+1}')
            axes[2, i].axis('off')
            
            # Add colorbar for difference
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def prepare_data_for_training(fmri_data: np.ndarray, 
                            stimulus_data: np.ndarray,
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1) -> Tuple[np.ndarray, ...]:
    """
    Prepare data for training by splitting into train/val/test sets.
    
    Args:
        fmri_data: fMRI data
        stimulus_data: Stimulus data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_fmri, train_stimulus, val_fmri, val_stimulus, test_fmri, test_stimulus)
    """
    n_samples = fmri_data.shape[0]
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Create random permutation
    indices = np.random.permutation(n_samples)
    
    # Split data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_fmri = fmri_data[train_indices]
    train_stimulus = stimulus_data[train_indices]
    
    val_fmri = fmri_data[val_indices]
    val_stimulus = stimulus_data[val_indices]
    
    test_fmri = fmri_data[test_indices]
    test_stimulus = stimulus_data[test_indices]
    
    return train_fmri, train_stimulus, val_fmri, val_stimulus, test_fmri, test_stimulus
