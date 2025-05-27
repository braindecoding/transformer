# Architecture Diagrams for Journal Publication

This document describes the professional architecture diagrams created for high-impact journal submissions.

## 📊 Available Diagrams

### 1. Main Brain Decoder Architecture
**File**: `brain_decoder_architecture.png/.pdf`
- **Purpose**: Complete system overview for main paper figure
- **Content**: 
  - fMRI input processing
  - Neural network architecture
  - Output generation
  - Evaluation metrics
  - Technical specifications
- **Recommended use**: Main figure in journal paper

### 2. Detailed Network Architecture  
**File**: `detailed_network_architecture.png/.pdf`
- **Purpose**: Neural network layer details
- **Content**:
  - Layer-by-layer breakdown
  - Parameter counts
  - Attention mechanisms
  - Performance metrics
- **Recommended use**: Supplementary material or methods section

### 3. Data Flow and Processing Pipeline
**File**: `data_flow_diagram.png/.pdf`
- **Purpose**: Complete processing pipeline
- **Content**:
  - 5-stage processing flow
  - Technical implementation details
  - Performance comparison table
  - Hardware requirements
- **Recommended use**: Methods section or supplementary material

### 4. Transformer Architecture Details
**File**: `transformer_architecture_detail.png/.pdf`
- **Purpose**: Detailed transformer components
- **Content**:
  - Encoder-decoder structure
  - Multi-head attention details
  - Mathematical formulations
  - Layer specifications
- **Recommended use**: Technical appendix or methods section

### 5. Comprehensive Evaluation Metrics
**File**: `evaluation_metrics_diagram.png/.pdf`
- **Purpose**: Complete evaluation overview
- **Content**:
  - All evaluation metrics
  - Quality assessment radar chart
  - Sample reconstructions
  - Training progress
- **Recommended use**: Results section

## 🎯 Journal Submission Guidelines

### High-Impact Journals (Nature, Science, Cell)
- **Main Figure**: Use `brain_decoder_architecture.pdf`
- **Resolution**: 300 DPI minimum
- **Format**: PDF for vector graphics
- **Size**: Designed for full-page or half-page layouts

### Specialized Journals (NeuroImage, IEEE, etc.)
- **Multiple Figures**: Use combination of diagrams
- **Technical Detail**: Include transformer and data flow diagrams
- **Supplementary**: Use detailed network architecture

### Conference Presentations
- **Format**: Use PNG versions for slides
- **Resolution**: 300 DPI for print, 150 DPI for screen
- **Layout**: All diagrams designed for 16:9 and 4:3 ratios

## 🚀 Quick Start

### Generate All Diagrams
```bash
# Run the diagram generator
python create_architecture_diagram.py

# Output will be in architecture_diagrams/ folder
```

### Individual Diagram Generation
```python
from create_architecture_diagram import *

# Generate specific diagram
fig = create_brain_decoder_architecture()
fig.savefig('my_diagram.pdf', dpi=300, bbox_inches='tight')
```

## 📝 Technical Specifications

### Design Standards
- **Resolution**: 300 DPI for publication quality
- **Color Scheme**: Professional, colorblind-friendly palette
- **Typography**: Clear, readable fonts (10-16pt)
- **Layout**: Balanced, hierarchical information flow

### File Formats
- **PDF**: Vector graphics for journal submission
- **PNG**: Raster graphics for presentations
- **Size**: Optimized for A4 and US Letter formats

### Content Standards
- **Accuracy**: All technical details verified
- **Completeness**: Includes all system components
- **Clarity**: Clear labels and annotations
- **Professional**: Publication-ready quality

## 🎨 Customization

### Color Scheme
```python
colors = {
    'input': '#2E86AB',      # Blue - Input processing
    'processing': '#A23B72',  # Purple - Neural network
    'output': '#F18F01',     # Orange - Output generation
    'evaluation': '#C73E1D',  # Red - Evaluation metrics
    'accent': '#27AE60'      # Green - Training/success
}
```

### Modifying Diagrams
1. Edit the corresponding function in `create_architecture_diagram.py`
2. Adjust colors, text, or layout as needed
3. Regenerate with `python create_architecture_diagram.py`

## 📊 Performance Metrics Included

All diagrams include current performance metrics:
- **Correlation**: 0.5969
- **PSNR**: ~20 dB
- **SSIM**: ~0.6
- **Model Parameters**: 1.9M
- **Training Time**: ~3 minutes
- **Dataset**: 62 train, 28 test samples

## 🔧 Dependencies

Required packages for diagram generation:
```bash
pip install matplotlib seaborn numpy
```

## 📚 Citation Guidelines

When using these diagrams in publications:

```bibtex
@article{brain_decoder_2024,
    title={Brain Decoder: Transformer-based Visual Stimulus Reconstruction from fMRI Signals},
    author={[Your Name]},
    journal={[Journal Name]},
    year={2024},
    note={Architecture diagrams generated with professional visualization toolkit}
}
```

## 🎯 Best Practices for Journal Submission

### Figure Preparation
1. **Use PDF format** for vector graphics
2. **300 DPI minimum** resolution
3. **Clear, readable text** at final print size
4. **Consistent color scheme** across all figures

### Caption Writing
- **Comprehensive**: Explain all components
- **Technical**: Include relevant specifications
- **Clear**: Accessible to broad audience
- **Structured**: Use consistent format

### Supplementary Material
- Include high-resolution versions
- Provide source code for reproducibility
- Add detailed technical specifications
- Include performance comparison tables

## 🔍 Quality Checklist

Before submission, verify:
- [ ] All text is readable at print size
- [ ] Colors are distinguishable (colorblind-friendly)
- [ ] Technical details are accurate
- [ ] Layout is balanced and professional
- [ ] File format matches journal requirements
- [ ] Resolution meets publication standards

## 📞 Support

For questions about diagram customization or technical details:
1. Check the source code in `create_architecture_diagram.py`
2. Review this documentation
3. Refer to the main project documentation

---

**🎨 Professional Architecture Diagrams**: Publication-ready visualizations for high-impact journal submissions.
