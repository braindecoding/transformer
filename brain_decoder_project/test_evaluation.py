"""
Test Comprehensive Evaluation System

This script tests the comprehensive evaluation system with synthetic data
to ensure all metrics and plots work correctly.

Author: AI Assistant
Date: 2024
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_synthetic_data(num_samples=10, image_size=28):
    """Create synthetic test data."""
    print("🔄 Creating synthetic test data...")
    
    # Create target images (simple patterns)
    targets = []
    predictions = []
    
    for i in range(num_samples):
        # Create target with simple pattern
        target = np.zeros((image_size, image_size))
        
        # Add some patterns
        if i % 3 == 0:
            # Circle pattern
            center = image_size // 2
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= (image_size // 4) ** 2
            target[mask] = 1.0
        elif i % 3 == 1:
            # Square pattern
            start = image_size // 4
            end = 3 * image_size // 4
            target[start:end, start:end] = 1.0
        else:
            # Diagonal pattern
            np.fill_diagonal(target, 1.0)
            np.fill_diagonal(np.fliplr(target), 1.0)
        
        # Create prediction with some noise
        prediction = target + np.random.normal(0, 0.1, target.shape)
        prediction = np.clip(prediction, 0, 1)
        
        targets.append(target.flatten())
        predictions.append(prediction.flatten())
    
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    print(f"✅ Created {num_samples} synthetic samples")
    print(f"   Target shape: {targets.shape}")
    print(f"   Prediction shape: {predictions.shape}")
    
    return predictions, targets

def test_basic_evaluation():
    """Test basic evaluation without advanced packages."""
    print("\n🧪 TESTING BASIC EVALUATION")
    print("=" * 50)
    
    # Create test data
    predictions, targets = create_synthetic_data()
    
    # Test basic metrics
    mse = np.mean((predictions - targets) ** 2)
    correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    
    print(f"✅ Basic metrics computed:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Correlation: {correlation:.4f}")
    
    # Test basic PSNR and SSIM
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        psnr_scores = []
        ssim_scores = []
        
        for i in range(len(predictions)):
            pred_img = predictions[i].reshape(28, 28)
            target_img = targets[i].reshape(28, 28)
            
            # Convert to uint8
            pred_uint8 = (pred_img * 255).astype(np.uint8)
            target_uint8 = (target_img * 255).astype(np.uint8)
            
            psnr_val = psnr(target_uint8, pred_uint8, data_range=255)
            ssim_val = ssim(target_uint8, pred_uint8, data_range=255)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
        
        print(f"✅ Image quality metrics computed:")
        print(f"   PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
        print(f"   SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
        
    except ImportError as e:
        print(f"⚠️  Image quality metrics not available: {e}")
    
    return predictions, targets

def test_comprehensive_evaluation():
    """Test comprehensive evaluation with all metrics."""
    print("\n🧪 TESTING COMPREHENSIVE EVALUATION")
    print("=" * 50)
    
    try:
        from brain_decoder.evaluation import EvaluationMetrics
        
        # Create test data
        predictions, targets = create_synthetic_data()
        
        # Initialize evaluator
        evaluator = EvaluationMetrics()
        
        # Run comprehensive evaluation
        print("🔄 Running comprehensive evaluation...")
        results = evaluator.evaluate_all(predictions, targets)
        
        # Print results
        evaluator.print_results(results)
        
        # Create plots
        print("🎨 Creating evaluation plots...")
        
        # Create output directory
        output_dir = "test_evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics plot
        evaluator.plot_metrics(results, 
                             save_path=os.path.join(output_dir, "test_metrics.png"),
                             show=False)
        
        # Sample reconstructions plot
        evaluator.plot_sample_reconstructions(
            predictions, targets,
            num_samples=min(6, len(predictions)),
            save_path=os.path.join(output_dir, "test_reconstructions.png"),
            show=False
        )
        
        print(f"✅ Comprehensive evaluation completed!")
        print(f"📁 Test results saved to: {output_dir}/")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Comprehensive evaluation not available: {e}")
        print("💡 Run: python install_evaluation_deps.py")
        return False

def test_plotting():
    """Test basic plotting functionality."""
    print("\n🧪 TESTING PLOTTING FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Create test data
        predictions, targets = create_synthetic_data(num_samples=6)
        
        # Create a simple comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Test Evaluation - Sample Reconstructions', fontsize=14, fontweight='bold')
        
        for i in range(min(3, len(predictions))):
            # Target
            target_img = targets[i].reshape(28, 28)
            axes[0, i].imshow(target_img, cmap='gray')
            axes[0, i].set_title(f'Target {i+1}')
            axes[0, i].axis('off')
            
            # Prediction
            pred_img = predictions[i].reshape(28, 28)
            axes[1, i].imshow(pred_img, cmap='gray')
            axes[1, i].set_title(f'Predicted {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = "test_evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "test_basic_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Basic plotting test completed!")
        print(f"📁 Test plot saved to: {output_dir}/test_basic_plot.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        return False

def main():
    """Run all evaluation tests."""
    print("🧠 BRAIN DECODER EVALUATION SYSTEM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test basic evaluation
    basic_success = True
    try:
        test_basic_evaluation()
    except Exception as e:
        print(f"❌ Basic evaluation test failed: {e}")
        basic_success = False
    
    # Test plotting
    plot_success = test_plotting()
    
    # Test comprehensive evaluation
    comprehensive_success = test_comprehensive_evaluation()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if basic_success:
        print("✅ Basic evaluation: PASSED")
    else:
        print("❌ Basic evaluation: FAILED")
    
    if plot_success:
        print("✅ Basic plotting: PASSED")
    else:
        print("❌ Basic plotting: FAILED")
    
    if comprehensive_success:
        print("✅ Comprehensive evaluation: PASSED")
    else:
        print("⚠️  Comprehensive evaluation: NOT AVAILABLE")
        print("💡 Install dependencies with: python install_evaluation_deps.py")
    
    print("\n🎯 Next steps:")
    if not comprehensive_success:
        print("1. Install evaluation dependencies: python install_evaluation_deps.py")
        print("2. Re-run this test: python test_evaluation.py")
        print("3. Run actual evaluation: python src/brain_decoder_main.py")
    else:
        print("1. Run actual evaluation: python src/brain_decoder_main.py")
        print("2. Or use standalone evaluation: python scripts/comprehensive_evaluation.py")
    
    print(f"\n✅ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
