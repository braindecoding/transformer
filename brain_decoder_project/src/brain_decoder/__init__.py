"""
Brain Decoder Package

Transformer-based fMRI to Stimulus Reconstruction System
"""

from .model import BrainDecoder, BrainDecoderLoss
from .trainer import BrainDecoderTrainer, BrainDecodingDataset
from .utils import BrainDecoderEvaluator, BrainDecoderVisualizer, prepare_data_for_training

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    'BrainDecoder',
    'BrainDecoderLoss', 
    'BrainDecoderTrainer',
    'BrainDecodingDataset',
    'BrainDecoderEvaluator',
    'BrainDecoderVisualizer',
    'prepare_data_for_training'
]
