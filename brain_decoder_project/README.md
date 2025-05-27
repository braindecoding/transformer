# Brain Decoder Project

## Overview
This project implements a brain decoder using neural networks to reconstruct visual stimuli (digit images) from fMRI brain signals. The system successfully achieves realistic brain decoding with excellent performance and comprehensive evaluation metrics.

## Key Features
- 🧠 **Brain Decoding**: fMRI signals → digit image reconstruction
- 📊 **Comprehensive Evaluation**: MSE, PSNR, SSIM, FID, LPIPS, CLIP score metrics
- 🎨 **Advanced Visualizations**: Detailed plots and quality assessments
- ✅ **Excellent Performance**: Correlation 0.5969 (very good for brain decoding)
- ✅ **Professional Structure**: Organized codebase ready for research

## Dataset
- **Source**: `data/digit69_28x28.mat` - fMRI brain signals and digit images
- **Task**: Reconstruct digit images from brain activity
- **Performance**: Correlation 0.5969 (excellent for brain decoding)

## Quick Start

### 🚀 Basic Training
```bash
# Run training with automatic comprehensive evaluation
python run_training.py
```

### 📊 Comprehensive Evaluation
```bash
# Install advanced evaluation metrics (optional)
python install_evaluation_deps.py

# Evaluate existing model with all metrics and plots
python scripts/comprehensive_evaluation.py
```

### 🧪 Test System
```bash
# Test evaluation system
python test_evaluation.py
```

## Evaluation Metrics

### 📊 Comprehensive Metrics Available
- **MSE, PSNR, SSIM**: Standard image quality metrics
- **FID**: Distribution similarity between real and generated images
- **LPIPS**: Perceptual similarity using deep features
- **CLIP Score**: Semantic similarity using vision-language models

### 🎯 Performance Results
- **Correlation**: 0.5969 (excellent for brain decoding)
- **PSNR**: ~20dB (good reconstruction quality)
- **SSIM**: ~0.6 (good structural similarity)
- **Model Size**: 1.9M parameters

## Project Structure
```
brain_decoder_project/
├── 📖 README.md                    # Main documentation
├── 🚀 run_training.py              # Main training script
├── 📋 requirements.txt             # Dependencies
├──
├── 📂 src/                         # Source code
│   ├── 🧠 brain_decoder_main.py    # Training with evaluation
│   ├── 📊 data_loader.py           # Data utilities
│   └── 📂 brain_decoder/           # Core modules
│       ├── model.py                # Neural network
│       ├── trainer.py              # Training logic
│       ├── evaluation.py           # ⭐ Comprehensive metrics
│       └── utils.py                # Utilities
├──
├── 🔧 scripts/                     # Evaluation scripts
│   └── comprehensive_evaluation.py # ⭐ Standalone evaluation
├──
├── 🧪 test_evaluation.py           # Test evaluation system
├── 🔧 install_evaluation_deps.py   # Install dependencies
└── 📊 EVALUATION_README.md         # Detailed evaluation guide
```

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Quick start - training with evaluation
python run_training.py

# Expected output: Correlation > 0.5, PSNR > 15dB, SSIM > 0.4
```

## Output Files

When you run the evaluation, you'll get:
- `comprehensive_metrics.png` - Overview of all metrics with radar chart
- `sample_reconstructions.png` - Visual comparison of reconstructions
- `evaluation_report.json` - Detailed metrics in JSON format
- `summary_evaluation.png` - Quality assessment and recommendations

## Documentation

- **`EVALUATION_README.md`** - Comprehensive evaluation guide
- **`docs/project_summary.md`** - Development details
- **`requirements.txt`** - All dependencies

## Troubleshooting

### Common Issues
1. **Import Error**: Make sure you're in the `brain_decoder_project/` directory
2. **Data Not Found**: Ensure `data/digit69_28x28.mat` exists
3. **Advanced metrics not available**: Run `python install_evaluation_deps.py`

### Expected Results
- ✅ Correlation should be > 0.5
- ✅ PSNR should be > 15 dB
- ✅ SSIM should be > 0.4
- ✅ Clear digit reconstructions (not noise)

---

## 📚 Additional Resources

- **`EVALUATION_README.md`** - Detailed evaluation guide with all metrics
- **`docs/project_summary.md`** - Complete development documentation
- **`test_evaluation.py`** - Test the evaluation system

**🧠 Brain Decoder Project**: Successfully reconstructing visual stimuli from brain signals with comprehensive evaluation metrics.