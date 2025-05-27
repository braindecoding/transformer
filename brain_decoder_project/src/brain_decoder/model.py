"""
Brain Decoder Model Architecture

This module implements the core transformer-based decoder for reconstructing 
visual stimuli from fMRI signals.

Components:
1. fMRI Encoder: Processes fMRI signals into feature representations
2. Transformer Decoder: Generates stimulus reconstructions  
3. Stimulus Decoder: Converts features back to stimulus space

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math
import warnings

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class fMRIEncoder(nn.Module):
    """Encoder for fMRI signals."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Temporal processing layers
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Spatial attention for voxel selection
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, fmri_data: torch.Tensor) -> torch.Tensor:
        """Forward pass for fMRI encoder."""
        # Project to hidden dimension
        x = self.input_projection(fmri_data)
        x = self.dropout(x)
        
        # Apply temporal processing
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Apply spatial attention
        x_attended, _ = self.spatial_attention(x, x, x)
        x = x + x_attended  # Residual connection
        
        # Final normalization
        x = self.layer_norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for stimulus reconstruction."""
    
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Cross-attention for fMRI-stimulus alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for transformer decoder."""
        # Add positional encoding
        tgt = self.pos_encoding(tgt)
        memory = self.pos_encoding(memory)
        
        # Apply transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Apply cross-attention
        cross_attended, _ = self.cross_attention(output, memory, memory)
        output = output + cross_attended
        
        # Final normalization
        output = self.layer_norm(output)
        
        return output


class StimulusDecoder(nn.Module):
    """Decoder for converting features back to stimulus space."""
    
    def __init__(self,
                 input_dim: int = 512,
                 output_shape: Tuple[int, ...] = (28, 28),
                 num_channels: int = 1,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.output_shape = output_shape
        self.num_channels = num_channels
        self.output_size = num_channels * np.prod(output_shape)
        
        # Build decoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, self.output_size))
        
        self.decoder = nn.Sequential(*layers)
        
        # Optional: Convolutional decoder for better spatial structure
        if len(output_shape) == 2:  # 2D images
            self.use_conv = True
            self.conv_decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, num_channels, 4, 2, 1),
                nn.Sigmoid()
            )
        else:
            self.use_conv = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for stimulus decoder."""
        batch_size, seq_len, _ = x.shape
        
        if self.use_conv:
            # Use convolutional decoder
            x = x.mean(dim=1)  # Average over time
            x = self.decoder[:-1](x)  # Apply all but last layer
            
            # Reshape to spatial dimensions for conv decoder
            spatial_size = int(np.sqrt(x.size(1) // 64))
            x = x.view(batch_size, 64, spatial_size, spatial_size)
            
            # Apply convolutional decoder
            x = self.conv_decoder(x)
            
        else:
            # Use fully connected decoder
            x = self.decoder(x)
            
            # Reshape to output format
            if len(self.output_shape) == 2:
                x = x.view(batch_size, seq_len, self.num_channels, 
                          self.output_shape[0], self.output_shape[1])
            else:
                x = x.view(batch_size, seq_len, *self.output_shape)
            
            # Apply sigmoid for image-like outputs
            x = torch.sigmoid(x)
        
        return x


class BrainDecoder(nn.Module):
    """Complete Brain Decoder model combining all components."""
    
    def __init__(self,
                 fmri_input_dim: int,
                 stimulus_shape: Tuple[int, ...] = (28, 28),
                 stimulus_channels: int = 1,
                 hidden_dim: int = 512,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 6,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.fmri_input_dim = fmri_input_dim
        self.stimulus_shape = stimulus_shape
        self.stimulus_channels = stimulus_channels
        self.hidden_dim = hidden_dim
        
        # Components
        self.fmri_encoder = fMRIEncoder(
            input_dim=fmri_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        self.transformer_decoder = TransformerDecoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.stimulus_decoder = StimulusDecoder(
            input_dim=hidden_dim,
            output_shape=stimulus_shape,
            num_channels=stimulus_channels
        )
        
        # Learnable stimulus query tokens
        self.stimulus_queries = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        
    def forward(self, 
                fmri_data: torch.Tensor,
                target_length: Optional[int] = None) -> torch.Tensor:
        """Forward pass for complete brain decoder."""
        batch_size = fmri_data.size(0)
        
        # Encode fMRI signals
        fmri_features = self.fmri_encoder(fmri_data)
        
        # Prepare target sequence (stimulus queries)
        if target_length is None:
            target_length = fmri_data.size(1)
        
        # Expand stimulus queries for batch and sequence length
        tgt = self.stimulus_queries.expand(batch_size, target_length, -1)
        
        # Apply transformer decoder
        decoded_features = self.transformer_decoder(
            tgt=tgt,
            memory=fmri_features
        )
        
        # Decode to stimulus space
        reconstructed_stimuli = self.stimulus_decoder(decoded_features)
        
        return reconstructed_stimuli


class BrainDecoderLoss(nn.Module):
    """Combined loss function for brain decoder training."""
    
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 consistency_weight: float = 0.05):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.consistency_weight = consistency_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, 
                predicted: torch.Tensor,
                target: torch.Tensor,
                fmri_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses = {}
        
        # Reconstruction loss (MSE + L1)
        mse_loss = self.mse_loss(predicted, target)
        l1_loss = self.l1_loss(predicted, target)
        reconstruction_loss = mse_loss + 0.1 * l1_loss
        
        losses['reconstruction'] = reconstruction_loss
        losses['mse'] = mse_loss
        losses['l1'] = l1_loss
        
        # Perceptual loss (simplified - using gradient similarity)
        if self.perceptual_weight > 0:
            pred_grad_x = torch.abs(predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
            pred_grad_y = torch.abs(predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
            
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            
            perceptual_loss = (self.mse_loss(pred_grad_x, target_grad_x) + 
                             self.mse_loss(pred_grad_y, target_grad_y))
            
            losses['perceptual'] = perceptual_loss
        else:
            losses['perceptual'] = torch.tensor(0.0, device=predicted.device)
        
        # Temporal consistency loss
        if self.consistency_weight > 0 and predicted.size(1) > 1:
            pred_diff = predicted[:, 1:] - predicted[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            consistency_loss = self.mse_loss(pred_diff, target_diff)
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=predicted.device)
        
        # Total loss
        total_loss = (self.reconstruction_weight * losses['reconstruction'] +
                     self.perceptual_weight * losses['perceptual'] +
                     self.consistency_weight * losses['consistency'])
        
        losses['total'] = total_loss
        
        return losses
