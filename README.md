# Brain Decoder Project

## Quick Start

The main project is located in the `brain_decoder_project/` folder.

```bash
cd brain_decoder_project/
python run_training.py
```

## Project Structure

```
transformer/
├── README.md                    # This file
└── brain_decoder_project/       # 🎯 MAIN PROJECT FOLDER
    ├── README.md                # 📖 Complete documentation
    ├── run_training.py          # 🚀 Main training script
    ├── src/                     # 📂 Source code
    ├── models/                  # 🤖 Trained models
    ├── results/                 # 📊 Training results
    ├── data/                    # 💾 Dataset
    ├── docs/                    # 📚 Documentation
    └── scripts/                 # 🔧 Utility scripts
```

## Key Results

- **🏆 Best Model**: `brain_decoder_project/models/brain_decoder_final.pth`
- **📈 Performance**: Correlation 0.5969 (excellent for brain decoding)
- **🎨 Visualization**: `brain_decoder_project/results/final_training_results.png`
- **📊 Dataset**: 90 samples of fMRI + digit images

## What This Project Does

**Brain Decoder**: Reconstructs digit images from fMRI brain signals using neural networks.

```
fMRI Brain Signals → Neural Network → Reconstructed Digit Images
     (Input)            (AI Model)         (Output)
```

## Documentation

See `brain_decoder_project/README.md` for complete documentation including:
- Installation instructions
- Usage examples
- Technical details
- Performance metrics
- Troubleshooting guide

## Quick Commands

```bash
# Navigate to project
cd brain_decoder_project/

# Install dependencies
pip install -r requirements.txt

# Run training
python run_training.py

# Expected: Correlation ~0.60, clear digit reconstruction
```

---

**🧠 Brain Decoder**: Successfully reconstructing visual stimuli from brain signals using neural networks.
