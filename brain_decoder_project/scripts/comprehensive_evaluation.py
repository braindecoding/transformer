"""
Comprehensive Evaluation Script for Brain Decoder

This script provides comprehensive evaluation of brain decoder models using
all available metrics (MSE, PSNR, SSIM, FID, LPIPS, CLIP score) and generates
detailed plots and reports.

Usage:
    python comprehensive_evaluation.py --model_path path/to/model.pth --data_path path/to/data

Author: AI Assistant
Date: 2024
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json

from brain_decoder.evaluation import EvaluationMetrics
from brain_decoder_main import load_correct_data, BrainDecoder
from data_loader import DataLoader as NeuroDataLoader


def load_model(model_path: str, fmri_dim: int, stimulus_dim: int):
    """Load a trained brain decoder model."""
    print(f"🔄 Loading model from: {model_path}")
    
    # Create model architecture
    model = BrainDecoder(fmri_dim=fmri_dim, stimulus_dim=stimulus_dim)
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def evaluate_model(model, test_fmri, test_stimulus, device='cpu'):
    """Evaluate model on test data."""
    print("🔄 Running model inference...")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        test_fmri_tensor = torch.FloatTensor(test_fmri).to(device)
        predictions = model(test_fmri_tensor)
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        target_np = test_stimulus
        
    print("✅ Model inference completed!")
    return predictions_np, target_np


def save_evaluation_report(results: dict, save_dir: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, list):
            json_results[key] = value
        else:
            json_results[key] = float(value) if not isinstance(value, str) else value
    
    # Add metadata
    json_results['evaluation_timestamp'] = datetime.now().isoformat()
    json_results['evaluation_script'] = 'comprehensive_evaluation.py'
    
    # Save to file
    report_path = os.path.join(save_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📄 Evaluation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Brain Decoder Evaluation')
    parser.add_argument('--model_path', type=str, default='correct_results/correct_brain_decoder.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='Number of sample reconstructions to plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧠 COMPREHENSIVE BRAIN DECODER EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\n🔄 Loading data...")
    result = load_correct_data()
    if result[0] is None:
        print("❌ Failed to load data")
        return
    
    fmri_data, stimulus_data, labels, n_train_samples, n_test_samples = result
    
    # Use test split
    test_fmri = fmri_data[n_train_samples:]
    test_stimulus = stimulus_data[n_train_samples:]
    
    print(f"✅ Data loaded successfully!")
    print(f"   Test samples: {len(test_fmri)}")
    print(f"   fMRI shape: {test_fmri.shape}")
    print(f"   Stimulus shape: {test_stimulus.shape}")
    
    # Load model
    print("\n🔄 Loading model...")
    model = load_model(args.model_path, fmri_data.shape[1], stimulus_data.shape[1])
    if model is None:
        return
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    predictions, targets = evaluate_model(model, test_fmri, test_stimulus, device)
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics(device=str(device))
    
    # Compute all metrics
    print("\n📊 Computing comprehensive metrics...")
    results = evaluator.evaluate_all(predictions, targets)
    
    # Print results
    evaluator.print_results(results)
    
    # Create plots
    print("\n🎨 Creating visualizations...")
    
    # 1. Metrics plot
    metrics_plot_path = os.path.join(args.output_dir, 'comprehensive_metrics.png')
    evaluator.plot_metrics(results, save_path=metrics_plot_path, show=False)
    
    # 2. Sample reconstructions plot
    samples_plot_path = os.path.join(args.output_dir, 'sample_reconstructions.png')
    evaluator.plot_sample_reconstructions(
        predictions, targets, 
        num_samples=args.num_samples,
        save_path=samples_plot_path, 
        show=False
    )
    
    # 3. Training curve comparison (if available)
    training_plot_path = os.path.join(args.output_dir, 'training_comparison.png')
    create_training_comparison_plot(training_plot_path)
    
    # Save evaluation report
    print("\n📄 Saving evaluation report...")
    save_evaluation_report(results, args.output_dir)
    
    # Create summary plot
    print("\n🎨 Creating summary visualization...")
    create_summary_plot(results, predictions, targets, args.output_dir)
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"   - comprehensive_metrics.png")
    print(f"   - sample_reconstructions.png")
    print(f"   - evaluation_report.json")
    print(f"   - summary_evaluation.png")


def create_training_comparison_plot(save_path: str):
    """Create a comparison plot of training results if available."""
    try:
        # Look for existing training results
        training_results_paths = [
            'correct_results/correct_training_results.png',
            'results/final_training_results.png',
            'results/simple_training_results.png'
        ]
        
        fig, axes = plt.subplots(1, len(training_results_paths), figsize=(15, 5))
        if len(training_results_paths) == 1:
            axes = [axes]
        
        fig.suptitle('Training Results Comparison', fontsize=16, fontweight='bold')
        
        for i, path in enumerate(training_results_paths):
            if os.path.exists(path):
                # Load and display existing training plot
                img = plt.imread(path)
                axes[i].imshow(img)
                axes[i].set_title(os.path.basename(path).replace('_', ' ').title())
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Training plot\nnot found:\n{path}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Missing: {os.path.basename(path)}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training comparison plot saved to: {save_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create training comparison plot: {e}")


def create_summary_plot(results: dict, predictions: np.ndarray, targets: np.ndarray, output_dir: str):
    """Create a comprehensive summary plot."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a complex layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Brain Decoder Comprehensive Evaluation Summary', fontsize=20, fontweight='bold')
    
    # 1. Metrics summary table
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    
    metrics_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['MSE', f"{results['mse']:.6f}", 'Lower is better'],
        ['PSNR', f"{results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB", 'Higher is better'],
        ['SSIM', f"{results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}", 'Higher is better (0-1)'],
        ['FID', f"{results['fid']:.2f}", 'Lower is better'],
        ['LPIPS', f"{results['lpips']:.4f}", 'Lower is better'],
        ['CLIP Score', f"{results['clip_score']:.4f}", 'Higher is better (0-1)'],
        ['Correlation', f"{results['correlation']:.4f}", 'Higher is better (-1 to 1)']
    ]
    
    table = ax1.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax1.set_title('Evaluation Metrics Summary', fontweight='bold', pad=20)
    
    # 2. Best and worst reconstructions
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Find best and worst samples based on SSIM
    ssim_scores = results['ssim_scores']
    best_idx = np.argmax(ssim_scores)
    worst_idx = np.argmin(ssim_scores)
    
    # Plot best and worst
    for i, (idx, label) in enumerate([(best_idx, 'Best'), (worst_idx, 'Worst')]):
        if predictions[idx].ndim == 1:
            size = int(np.sqrt(len(predictions[idx])))
            pred_img = predictions[idx].reshape(size, size)
            target_img = targets[idx].reshape(size, size)
        else:
            pred_img = predictions[idx]
            target_img = targets[idx]
        
        # Create subplot for this comparison
        ax_sub = plt.subplot(2, 4, i*4 + 3)
        ax_sub.imshow(target_img, cmap='gray')
        ax_sub.set_title(f'{label} Target\nSSIM: {ssim_scores[idx]:.3f}')
        ax_sub.axis('off')
        
        ax_sub = plt.subplot(2, 4, i*4 + 4)
        ax_sub.imshow(pred_img, cmap='gray')
        ax_sub.set_title(f'{label} Predicted')
        ax_sub.axis('off')
    
    # 3. Metrics distribution plots
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.hist(results['psnr_scores'], bins=10, alpha=0.7, color='blue', label='PSNR')
    ax3.set_xlabel('PSNR (dB)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('PSNR Distribution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.hist(results['ssim_scores'], bins=10, alpha=0.7, color='green', label='SSIM')
    ax4.set_xlabel('SSIM')
    ax4.set_ylabel('Frequency')
    ax4.set_title('SSIM Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 4. Overall quality assessment
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create quality assessment text
    quality_text = f"""
    OVERALL QUALITY ASSESSMENT:
    
    📊 Reconstruction Quality: {'Excellent' if results['psnr_mean'] > 25 else 'Good' if results['psnr_mean'] > 20 else 'Fair' if results['psnr_mean'] > 15 else 'Poor'}
    🎯 Structural Similarity: {'Excellent' if results['ssim_mean'] > 0.8 else 'Good' if results['ssim_mean'] > 0.6 else 'Fair' if results['ssim_mean'] > 0.4 else 'Poor'}
    🎨 Perceptual Quality: {'Good' if results['lpips'] < 0.3 else 'Fair' if results['lpips'] < 0.5 else 'Poor'}
    🧠 Semantic Similarity: {'Excellent' if results['clip_score'] > 0.8 else 'Good' if results['clip_score'] > 0.6 else 'Fair' if results['clip_score'] > 0.4 else 'Poor'}
    
    📈 Key Insights:
    • Average PSNR of {results['psnr_mean']:.1f}dB indicates {'high' if results['psnr_mean'] > 20 else 'moderate' if results['psnr_mean'] > 15 else 'low'} signal quality
    • SSIM of {results['ssim_mean']:.3f} shows {'strong' if results['ssim_mean'] > 0.7 else 'moderate' if results['ssim_mean'] > 0.5 else 'weak'} structural preservation
    • Correlation of {results['correlation']:.3f} indicates {'strong' if abs(results['correlation']) > 0.7 else 'moderate' if abs(results['correlation']) > 0.5 else 'weak'} linear relationship
    
    🎯 Recommendations:
    {'• Model performs well across all metrics' if results['ssim_mean'] > 0.6 and results['psnr_mean'] > 20 else '• Consider improving perceptual quality' if results['lpips'] > 0.4 else '• Focus on structural similarity enhancement' if results['ssim_mean'] < 0.5 else '• Overall performance needs improvement'}
    """
    
    ax5.text(0.05, 0.95, quality_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax5.axis('off')
    
    # Save summary plot
    summary_path = os.path.join(output_dir, 'summary_evaluation.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary evaluation plot saved to: {summary_path}")


if __name__ == "__main__":
    main()
