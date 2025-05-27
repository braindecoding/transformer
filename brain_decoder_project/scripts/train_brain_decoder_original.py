"""
Simple Training Script for Brain Decoder

Cara penggunaan:
1. Siapkan data fMRI dan stimulus dalam folder 'data/'
2. Jalankan: python train_brain_decoder.py
3. Hasil akan disimpan di folder 'results/'

Author: AI Assistant
Date: 2024
"""

import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from datetime import datetime

# Import brain decoder modules
from brain_decoder import BrainDecoder, BrainDecoderTrainer, BrainDecodingDataset
from brain_decoder import BrainDecoderEvaluator, BrainDecoderVisualizer, prepare_data_for_training
from data_loader import DataLoader as NeuroDataLoader


def load_your_data(data_dir: str = "data"):
    """
    Load dan prepare data fMRI dan stimulus.

    PENTING: Sesuaikan fungsi ini dengan format data Anda!

    Returns:
        fmri_data: numpy array shape (n_samples, n_voxels)
        stimulus_data: numpy array shape (n_samples, height, width)
    """
    print("🔄 Loading data...")

    loader = NeuroDataLoader(data_dir)
    files = loader.list_files()

    if not files:
        raise FileNotFoundError(f"Tidak ada file di folder {data_dir}")

    print(f"📁 Ditemukan {len(files)} file:")
    for file in files:
        print(f"   - {file.name}")

    # Contoh: Load file MATLAB
    mat_files = [f for f in files if f.suffix.lower() == '.mat']

    if mat_files:
        result = loader.load_data(mat_files[0].name)
        print(f"✅ Loaded {mat_files[0].name}")

        # SESUAIKAN BAGIAN INI DENGAN DATA ANDA!
        # Contoh untuk data digit 28x28
        data_dict = result['data']

        if isinstance(data_dict, dict):
            # Ambil variabel pertama sebagai contoh
            first_var = list(data_dict.keys())[0]
            raw_data = data_dict[first_var]

            print(f"📊 Data shape: {raw_data.shape}")

            # Handle berbagai format data
            if len(raw_data.shape) == 2:
                n_samples, n_features = raw_data.shape
                print(f"📊 Data: {n_samples} samples, {n_features} features")

                # Case 1: Data 784 features (28x28 images)
                if n_features == 784:
                    # Stimulus data (gambar 28x28)
                    stimulus_data = raw_data.reshape(n_samples, 28, 28)
                    stimulus_data = (stimulus_data - stimulus_data.min()) / (stimulus_data.max() - stimulus_data.min())

                    # fMRI data (simulasi)
                    n_voxels = 500
                    fmri_data = np.random.randn(n_samples, n_voxels)

                    # Tambahkan korelasi dengan stimulus
                    stimulus_flat = stimulus_data.reshape(n_samples, -1)
                    for i in range(min(n_voxels, stimulus_flat.shape[1])):
                        fmri_data[:, i] += stimulus_flat[:, i] * 0.3

                # Case 2: Data dengan features lain (seperti 3092)
                else:
                    print(f"🔄 Menggunakan data asli sebagai STIMULUS dengan {n_features} features")

                    # PERBAIKAN: Gunakan data asli sebagai STIMULUS, bukan fMRI
                    stimulus_size = 28
                    stimulus_features = stimulus_size * stimulus_size  # 784

                    # Ambil 784 features pertama dari data untuk stimulus
                    if n_features >= stimulus_features:
                        stimulus_flat = raw_data[:, :stimulus_features]
                    else:
                        # Jika features kurang dari 784, tile untuk mencapai 784
                        repeat_factor = (stimulus_features // n_features) + 1
                        stimulus_flat = np.tile(raw_data, (1, repeat_factor))[:, :stimulus_features]

                    # Reshape ke format gambar 28x28
                    stimulus_data = stimulus_flat.reshape(n_samples, stimulus_size, stimulus_size)

                    # Buat fMRI data sintetis berdasarkan stimulus
                    n_voxels = min(500, n_features)  # Maksimal 500 voxels untuk efisiensi
                    fmri_data = np.random.randn(n_samples, n_voxels) * 0.5

                    # Tambahkan korelasi dengan stimulus
                    for i in range(min(n_voxels, stimulus_features)):
                        fmri_data[:, i % n_voxels] += stimulus_flat[:, i] * 0.3

                    print(f"📊 Stimulus dibuat dari {stimulus_features} features pertama")
                    print(f"📊 fMRI sintetis dibuat dengan {n_voxels} voxels")

                print(f"✅ fMRI shape: {fmri_data.shape}")
                print(f"✅ Stimulus shape: {stimulus_data.shape}")

                return fmri_data, stimulus_data

            else:
                raise ValueError(f"Format data tidak dikenali: {raw_data.shape}. Expected 2D array.")

        else:
            raise ValueError("Expected dictionary dalam file MATLAB")

    else:
        raise FileNotFoundError("Tidak ada file .mat ditemukan")


def create_model_config():
    """Konfigurasi model - sesuaikan dengan kebutuhan Anda."""
    return {
        # Model architecture (disesuaikan untuk data kecil)
        'hidden_dim': 64,            # Ukuran hidden layer (kecil untuk data 10 samples)
        'num_encoder_layers': 1,     # Jumlah layer encoder (kecil untuk demo)
        'num_decoder_layers': 2,     # Jumlah layer decoder (kecil untuk demo)
        'nhead': 4,                  # Jumlah attention heads (kecil untuk demo)
        'dropout': 0.1,              # Dropout rate (0.1-0.3)

        # Training parameters (disesuaikan untuk data kecil)
        'batch_size': 2,             # Batch size kecil untuk 10 samples
        'learning_rate': 1e-3,       # Learning rate lebih besar untuk konvergensi cepat
        'num_epochs': 20,            # Epoch lebih sedikit untuk demo
        'sequence_length': 5,        # Sequence pendek untuk data kecil

        # Data parameters (disesuaikan untuk 10 samples)
        'train_ratio': 0.5,          # 5 samples untuk training
        'val_ratio': 0.3,            # 3 samples untuk validation
        # Test: 2 samples

        # Saving
        'save_every': 5,             # Save checkpoint setiap 5 epoch
        'early_stopping_patience': 10  # Early stopping patience
    }


def main():
    """Fungsi utama untuk training."""

    print("🧠 BRAIN DECODER TRAINING")
    print("=" * 50)

    # 1. Setup
    config = create_model_config()
    results_dir = f"results/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"📁 Results akan disimpan di: {results_dir}")

    try:
        # 2. Load data
        fmri_data, stimulus_data = load_your_data()

        # Normalize data
        fmri_data = (fmri_data - fmri_data.mean()) / fmri_data.std()
        stimulus_data = np.clip(stimulus_data, 0, 1)

        print(f"📊 Data summary:")
        print(f"   fMRI: {fmri_data.shape} (mean: {fmri_data.mean():.3f}, std: {fmri_data.std():.3f})")
        print(f"   Stimulus: {stimulus_data.shape} (min: {stimulus_data.min():.3f}, max: {stimulus_data.max():.3f})")

        # 3. Split data
        train_fmri, train_stimulus, val_fmri, val_stimulus, test_fmri, test_stimulus = \
            prepare_data_for_training(
                fmri_data, stimulus_data,
                config['train_ratio'], config['val_ratio']
            )

        print(f"\n📊 Data splits:")
        print(f"   Train: {train_fmri.shape[0]} samples")
        print(f"   Validation: {val_fmri.shape[0]} samples")
        print(f"   Test: {test_fmri.shape[0]} samples")

        # 4. Create datasets
        train_dataset = BrainDecodingDataset(
            train_fmri, train_stimulus,
            sequence_length=config['sequence_length']
        )

        val_dataset = BrainDecodingDataset(
            val_fmri, val_stimulus,
            sequence_length=config['sequence_length']
        )

        # 5. Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0  # Windows compatibility
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print(f"\n🔄 Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        # 6. Create model
        model = BrainDecoder(
            fmri_input_dim=fmri_data.shape[1],
            stimulus_shape=stimulus_data.shape[1:],
            stimulus_channels=1,
            hidden_dim=config['hidden_dim'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            nhead=config['nhead'],
            dropout=config['dropout']
        )

        print(f"\n🤖 Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {next(model.parameters()).device}")

        # 7. Create trainer
        trainer = BrainDecoderTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config['learning_rate']
        )

        # 8. Train model
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['learning_rate']}")

        history = trainer.train(
            num_epochs=config['num_epochs'],
            save_dir=os.path.join(results_dir, 'checkpoints'),
            save_every=config['save_every'],
            early_stopping_patience=config['early_stopping_patience']
        )

        # 9. Evaluate on test set
        print(f"\n📊 Evaluating on test set...")

        # Load best model
        best_model_path = os.path.join(results_dir, 'checkpoints', 'best_model.pth')
        if os.path.exists(best_model_path):
            trainer.load_checkpoint(best_model_path)

        # Test evaluation
        model.eval()
        test_dataset = BrainDecodingDataset(
            test_fmri, test_stimulus,
            sequence_length=config['sequence_length']
        )

        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for fmri_batch, stimulus_batch in test_loader:
                fmri_batch = fmri_batch.to(trainer.device)
                predictions = model(fmri_batch)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(stimulus_batch.numpy())

        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        evaluator = BrainDecoderEvaluator()
        metrics = evaluator.evaluate_reconstruction(predictions, targets)

        print(f"\n📈 Test Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

        # 10. Save results and create visualizations
        print(f"\n💾 Saving results...")

        # Save results
        results = {
            'config': config,
            'history': history,
            'test_metrics': metrics,
            'data_info': {
                'fmri_shape': fmri_data.shape,
                'stimulus_shape': stimulus_data.shape,
                'train_samples': train_fmri.shape[0],
                'val_samples': val_fmri.shape[0],
                'test_samples': test_fmri.shape[0]
            }
        }

        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create visualizations
        visualizer = BrainDecoderVisualizer()

        # Plot training history
        visualizer.plot_training_history(
            history,
            save_path=os.path.join(results_dir, 'training_history.png')
        )

        # Plot reconstruction examples
        visualizer.plot_reconstruction_comparison(
            predictions[:5], targets[:5],
            save_path=os.path.join(results_dir, 'reconstruction_examples.png')
        )

        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"\n📊 Final metrics:")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

        print(f"\n🔧 Troubleshooting tips:")
        print(f"1. Pastikan data fMRI dan stimulus sudah benar")
        print(f"2. Sesuaikan batch_size jika out of memory")
        print(f"3. Kurangi model size jika terlalu lambat")
        print(f"4. Check format data di fungsi load_your_data()")


if __name__ == "__main__":
    main()
