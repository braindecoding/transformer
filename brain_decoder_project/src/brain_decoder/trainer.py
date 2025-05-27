"""
Training Pipeline for Brain Decoder

This module provides training utilities, data preparation, and evaluation
for the transformer-based brain decoder.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from tqdm import tqdm
import json
from datetime import datetime

from .model import BrainDecoder, BrainDecoderLoss


class BrainDecodingDataset(Dataset):
    """Dataset class for brain decoding experiments."""

    def __init__(self,
                 fmri_data: np.ndarray,
                 stimulus_data: np.ndarray,
                 sequence_length: int = 50,
                 overlap: float = 0.5):
        """
        Initialize dataset.

        Args:
            fmri_data: fMRI data of shape (n_samples, n_voxels) or (n_samples, time, n_voxels)
            stimulus_data: Stimulus data of shape (n_samples, height, width) or (n_samples, time, height, width)
            sequence_length: Length of sequences for training
            overlap: Overlap between consecutive sequences
        """
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stimulus_data = torch.FloatTensor(stimulus_data)
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Ensure data has time dimension
        if self.fmri_data.ndim == 2:
            self.fmri_data = self.fmri_data.unsqueeze(1)  # Add time dimension: (samples, 1, voxels)
        if self.stimulus_data.ndim == 3:
            self.stimulus_data = self.stimulus_data.unsqueeze(1)  # Add time dimension: (samples, 1, H, W)

        # Create sequences
        self._create_sequences()

    def _create_sequences(self):
        """Create sequences from the data."""
        n_samples = self.fmri_data.size(0)

        # For small datasets, use individual samples as sequences
        if n_samples < self.sequence_length:
            print(f"⚠️  Dataset too small ({n_samples} samples), using individual samples")
            self.sequences = []
            for i in range(n_samples):
                # Use single sample, expand time dimension to sequence_length=1
                fmri_seq = self.fmri_data[i:i+1]  # (1, time, voxels)
                stimulus_seq = self.stimulus_data[i:i+1]  # (1, time, H, W)
                self.sequences.append((fmri_seq, stimulus_seq))
        else:
            # Original sequence creation for larger datasets
            step_size = int(self.sequence_length * (1 - self.overlap))
            self.sequences = []

            for i in range(0, n_samples - self.sequence_length + 1, step_size):
                end_idx = i + self.sequence_length

                fmri_seq = self.fmri_data[i:end_idx]
                stimulus_seq = self.stimulus_data[i:end_idx]

                self.sequences.append((fmri_seq, stimulus_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class BrainDecoderTrainer:
    """Trainer class for brain decoder model."""

    def __init__(self,
                 model: BrainDecoder,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = 'auto'):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = BrainDecoderLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'learning_rate': []
        }

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (fmri_data, stimulus_data) in enumerate(pbar):
            fmri_data = fmri_data.to(self.device)
            stimulus_data = stimulus_data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            predicted = self.model(fmri_data)

            # Compute loss
            losses = self.criterion(predicted, stimulus_data)

            # Backward pass
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'MSE': f"{losses['mse'].item():.4f}"
            })

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches

        return {'loss': avg_loss, 'mse': avg_mse}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {'loss': 0.0, 'mse': 0.0}

        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        with torch.no_grad():
            for fmri_data, stimulus_data in tqdm(self.val_loader, desc="Validation"):
                fmri_data = fmri_data.to(self.device)
                stimulus_data = stimulus_data.to(self.device)

                # Forward pass
                predicted = self.model(fmri_data)

                # Compute loss
                losses = self.criterion(predicted, stimulus_data)

                total_loss += losses['total'].item()
                total_mse += losses['mse'].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches

        return {'loss': avg_loss, 'mse': avg_mse}

    def train(self,
              num_epochs: int,
              save_dir: str = "checkpoints",
              save_every: int = 10,
              early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train MSE: {train_metrics['mse']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val MSE: {val_metrics['mse']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                # Save best model
                self.save_checkpoint(
                    os.path.join(save_dir, "best_model.pth"),
                    epoch,
                    val_metrics['loss']
                )
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                    epoch,
                    val_metrics['loss']
                )

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print("Training completed!")
        return self.history

    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

        print(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch'], checkpoint['val_loss']
