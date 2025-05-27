# Comprehensive Brain Decoder Evaluation System

This document describes the comprehensive evaluation system for the brain decoder project, which includes advanced metrics and visualization capabilities.

## 📊 Available Metrics

The evaluation system provides the following metrics:

### Basic Metrics
- **MSE (Mean Squared Error)**: Measures pixel-wise reconstruction error
- **Correlation**: Linear correlation between predicted and target images

### Image Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in dB
- **SSIM (Structural Similarity Index)**: Measures structural similarity (0-1 scale)

### Perceptual Metrics
- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual distance using deep features

### Semantic Metrics
- **CLIP Score**: Semantic similarity using CLIP embeddings

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install advanced evaluation dependencies
python install_evaluation_deps.py
```

### 2. Test the Evaluation System

```bash
# Test with synthetic data
python test_evaluation.py
```

### 3. Run Evaluation on Trained Model

```bash
# Using the main training script (includes evaluation)
python src/brain_decoder_main.py

# Or use standalone evaluation script
python scripts/comprehensive_evaluation.py --model_path correct_results/correct_brain_decoder.pth
```

## 📈 Output Files

The evaluation system generates several output files:

### Plots
- `comprehensive_metrics.png`: Overview of all metrics with distributions and radar chart
- `sample_reconstructions.png`: Visual comparison of target vs predicted images
- `summary_evaluation.png`: Comprehensive summary with quality assessment

### Data
- `evaluation_report.json`: Detailed metrics in JSON format for further analysis

## 🔧 Usage Examples

### Basic Usage in Code

```python
from brain_decoder.evaluation import EvaluationMetrics
import numpy as np

# Initialize evaluator
evaluator = EvaluationMetrics()

# Your predictions and targets (numpy arrays)
predictions = model_predictions  # Shape: (n_samples, height*width) or (n_samples, height, width)
targets = ground_truth          # Same shape as predictions

# Compute all metrics
results = evaluator.evaluate_all(predictions, targets)

# Print formatted results
evaluator.print_results(results)

# Create plots
evaluator.plot_metrics(results, save_path="metrics.png")
evaluator.plot_sample_reconstructions(predictions, targets, save_path="samples.png")
```

### Advanced Usage

```python
# Compute individual metrics
mse = evaluator.compute_mse(predictions, targets)
psnr_scores = evaluator.compute_psnr(predictions, targets)
ssim_scores = evaluator.compute_ssim(predictions, targets)
fid = evaluator.compute_fid(predictions, targets)
lpips = evaluator.compute_lpips(predictions, targets)
clip_score = evaluator.compute_clip_score(predictions, targets)
```

### Standalone Evaluation Script

```bash
# Evaluate a specific model
python scripts/comprehensive_evaluation.py \
    --model_path path/to/model.pth \
    --data_path path/to/data \
    --output_dir results \
    --num_samples 6
```

## 📊 Understanding the Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0 to ∞ dB (higher is better)
- **Good values**: >20 dB (excellent), >15 dB (good), >10 dB (fair)
- **Interpretation**: Measures reconstruction quality based on pixel differences

### SSIM (Structural Similarity Index)
- **Range**: -1 to 1 (higher is better)
- **Good values**: >0.8 (excellent), >0.6 (good), >0.4 (fair)
- **Interpretation**: Measures structural similarity considering luminance, contrast, and structure

### FID (Fréchet Inception Distance)
- **Range**: 0 to ∞ (lower is better)
- **Good values**: <50 (excellent), <100 (good), <200 (fair)
- **Interpretation**: Measures distribution similarity between real and generated images

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Range**: 0 to ∞ (lower is better)
- **Good values**: <0.3 (excellent), <0.5 (good), <0.7 (fair)
- **Interpretation**: Perceptual distance using deep neural network features

### CLIP Score
- **Range**: 0 to 1 (higher is better)
- **Good values**: >0.8 (excellent), >0.6 (good), >0.4 (fair)
- **Interpretation**: Semantic similarity using CLIP vision-language model

## 🎨 Visualization Features

### Comprehensive Metrics Plot
- Bar charts for basic metrics
- Histograms for PSNR and SSIM distributions
- Normalized perceptual metrics comparison
- CLIP score visualization
- Radar chart showing overall quality profile

### Sample Reconstructions Plot
- Side-by-side comparison of targets and predictions
- Difference maps with color coding
- Individual PSNR and SSIM scores for each sample

### Summary Evaluation Plot
- Metrics summary table
- Best and worst reconstruction examples
- Quality assessment and recommendations
- Distribution plots

## 🔧 Troubleshooting

### Common Issues

1. **Import Error for Advanced Metrics**
   ```
   Solution: Run `python install_evaluation_deps.py`
   ```

2. **CUDA Out of Memory**
   ```
   Solution: Use device='cpu' or reduce batch size
   ```

3. **CLIP Model Download Issues**
   ```
   Solution: Check internet connection or use fallback mode
   ```

### Fallback Modes

The system automatically falls back to simplified versions if advanced packages are not available:
- **LPIPS**: Uses gradient-based perceptual distance
- **CLIP**: Uses cosine similarity of normalized features
- **FID**: Uses simplified statistical distance

## 📝 Integration with Existing Code

The evaluation system is designed to integrate seamlessly with existing brain decoder training:

1. **Automatic Integration**: The main training script (`brain_decoder_main.py`) automatically uses comprehensive evaluation if available
2. **Backward Compatibility**: Falls back to basic metrics if advanced packages are not installed
3. **Minimal Code Changes**: Existing code continues to work without modifications

## 🎯 Best Practices

1. **Always evaluate on test set**: Use data not seen during training
2. **Multiple metrics**: Don't rely on a single metric; use the comprehensive suite
3. **Visual inspection**: Always look at sample reconstructions alongside metrics
4. **Baseline comparison**: Compare against simple baselines (e.g., mean image)
5. **Statistical significance**: Report confidence intervals for metrics

## 📚 References

- PSNR/SSIM: Wang et al., "Image quality assessment: from error visibility to structural similarity"
- FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
- LPIPS: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
