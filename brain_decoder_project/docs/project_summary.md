# Brain Decoder Project Summary

## Project Goal
Develop a neural network that can decode visual stimuli (digit images) from fMRI brain signals.

## Data Analysis
- **File**: digit69_28x28.mat
- **Variables**: stimTrn, fmriTrn, stimTest, fmriTest, labelTrn, labelTest
- **Correct Usage**: stimTrn (target), fmriTrn (input)

## Development Process
1. **Initial Attempts**: Used wrong variables, got noise as target
2. **Data Analysis**: Discovered correct stimulus/fMRI variables
3. **Final Solution**: Used stimTrn as target, fmriTrn as input
4. **Result**: Clear digit reconstruction, not noise

## Performance Metrics
- **Final Correlation**: 0.5969
- **Training Convergence**: Excellent (0.100 → 0.011)
- **Generalization**: Good (test loss 0.062)
- **Visual Quality**: Clear digit reconstruction

## Key Achievements
- Solved "noise target" problem
- Achieved realistic brain decoding
- Excellent model performance
- Clear visualization results

## Files Organization
- **Main Code**: src/brain_decoder_main.py
- **Final Model**: models/brain_decoder_final.pth
- **Best Results**: results/final_training_results.png
- **Documentation**: docs/ folder
- **Utilities**: scripts/ folder
