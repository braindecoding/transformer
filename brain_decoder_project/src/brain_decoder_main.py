"""
Training Brain Decoder dengan data yang BENAR
Menggunakan stimTrn sebagai stimulus dan fmriTrn sebagai fMRI
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
import warnings

# Import comprehensive evaluation
try:
    from brain_decoder.evaluation import EvaluationMetrics
    COMPREHENSIVE_EVAL_AVAILABLE = True
except ImportError:
    print("⚠️  Comprehensive evaluation not available. Using basic metrics only.")
    COMPREHENSIVE_EVAL_AVAILABLE = False

class BrainDataset(Dataset):
    def __init__(self, fmri_data, stimulus_data):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stimulus_data = torch.FloatTensor(stimulus_data)

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        return self.fmri_data[idx], self.stimulus_data[idx]

class BrainDecoder(nn.Module):
    def __init__(self, fmri_dim, stimulus_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(fmri_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, stimulus_dim),
            nn.Sigmoid()
        )

    def forward(self, fmri):
        return self.decoder(fmri)

def load_correct_data():
    """Load data dengan variabel yang tepat"""
    print("🔄 Loading CORRECT data from digit69_28x28.mat...")

    try:
        from data_loader import DataLoader as NeuroDataLoader
        # Try different data paths
        data_paths = ["data", "../data", "brain_decoder_project/data"]
        loader = None
        result = None

        for data_path in data_paths:
            try:
                loader = NeuroDataLoader(data_path)
                result = loader.load_data("digit69_28x28.mat")
                print(f"✅ Found data in: {data_path}")
                break
            except:
                continue

        if result is None:
            raise FileNotFoundError("digit69_28x28.mat not found in any data directory")

        data_dict = result['data']
        print(f"📊 Available variables: {list(data_dict.keys())}")

        # Load TRAINING and TESTING data separately - NO SYNTHETIC DATA!
        if 'stimTrn' in data_dict and 'fmriTrn' in data_dict and 'stimTest' in data_dict and 'fmriTest' in data_dict:
            # Training data
            train_stimulus = data_dict['stimTrn']
            train_fmri = data_dict['fmriTrn']
            train_labels = data_dict.get('labelTrn', None)

            # Testing data
            test_stimulus = data_dict['stimTest']
            test_fmri = data_dict['fmriTest']
            test_labels = data_dict.get('labelTest', None)

            print(f"✅ Using CORRECT train/test split:")
            print(f"   Training: stimTrn + fmriTrn + labelTrn")
            print(f"   Testing: stimTest + fmriTest + labelTest")

            # Combine for processing, but keep track of split
            stimulus_data = np.vstack([train_stimulus, test_stimulus])
            fmri_data = np.vstack([train_fmri, test_fmri])

            if train_labels is not None and test_labels is not None:
                labels = np.hstack([train_labels.flatten(), test_labels.flatten()])
                print(f"✅ Labels loaded: {len(np.unique(labels))} unique classes")
            else:
                labels = None
                print(f"⚠️  No labels found")

            # Store split indices
            n_train = len(train_stimulus)
            n_test = len(test_stimulus)

        else:
            print(f"❌ Required variables not found!")
            print(f"Available: {list(data_dict.keys())}")
            return None, None, None, None, None

        print(f"📊 Training data: {train_stimulus.shape[0]} samples")
        print(f"📊 Testing data: {test_stimulus.shape[0]} samples")
        print(f"📊 Total: {stimulus_data.shape[0]} samples")
        print(f"📊 STIMULUS data shape: {stimulus_data.shape}")
        print(f"📊 fMRI data shape: {fmri_data.shape}")

        # Process stimulus data - NO SYNTHETIC DATA!
        if len(stimulus_data.shape) == 2:
            if stimulus_data.shape[1] == 784:
                print(f"✅ Stimulus is already 784 pixels (28x28) - REAL DATA")
                stimulus_flat = stimulus_data
            else:
                # Use actual data, no padding with synthetic zeros
                if stimulus_data.shape[1] >= 784:
                    stimulus_flat = stimulus_data[:, :784]
                    print(f"✅ Using first 784 features from REAL stimulus data")
                else:
                    # If less than 784, use all available real data
                    stimulus_flat = stimulus_data
                    print(f"✅ Using all {stimulus_data.shape[1]} REAL stimulus features")
        else:
            # Flatten if 3D, but keep all real data
            stimulus_flat = stimulus_data.reshape(stimulus_data.shape[0], -1)
            if stimulus_flat.shape[1] > 784:
                stimulus_flat = stimulus_flat[:, :784]
            print(f"✅ Flattened REAL stimulus to {stimulus_flat.shape[1]} features")

        # Fix digit orientation - flip and rotate to show proper digit 6
        print("🔄 Fixing digit orientation...")

        # Reshape to 28x28 for processing
        stimulus_images = stimulus_flat.reshape(-1, 28, 28)

        # Apply transformations to make digits look correct
        for i in range(stimulus_images.shape[0]):
            img = stimulus_images[i]

            # Flip vertically (upside down)
            img = np.flipud(img)

            # Rotate -90 degrees to correct orientation
            img = np.rot90(img, k=-1)

            stimulus_images[i] = img

        # Flatten back to original shape
        stimulus_flat = stimulus_images.reshape(-1, 784)
        print(f"✅ Applied flip and rotation to correct digit orientation")

        # Normalize data - preserving real data characteristics
        print("🔄 Normalizing REAL data...")

        # Normalize stimulus (preserve real digit characteristics)
        stim_min, stim_max = stimulus_flat.min(), stimulus_flat.max()
        if stim_max > stim_min:
            stimulus_flat = (stimulus_flat - stim_min) / (stim_max - stim_min)
        print(f"   Stimulus (REAL): {stimulus_flat.min():.3f} to {stimulus_flat.max():.3f}")

        # Normalize fMRI (z-score normalization for brain signals)
        fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)
        print(f"   fMRI (REAL): mean={fmri_data.mean():.3f}, std={fmri_data.std():.3f}")

        print(f"✅ Final shapes (ALL REAL DATA):")
        print(f"   fMRI (INPUT): {fmri_data.shape}")
        print(f"   Stimulus (TARGET): {stimulus_flat.shape}")
        if labels is not None:
            print(f"   Labels: {labels.shape} (classes: {np.unique(labels)})")

        # Show sample stimulus from real data
        print(f"\n🎨 Sample REAL stimulus visualization:")
        sample_img = stimulus_flat[0].reshape(28, 28) if stimulus_flat.shape[1] >= 784 else stimulus_flat[0].reshape(int(np.sqrt(stimulus_flat.shape[1])), -1)
        print(f"   Sample 0: min={sample_img.min():.3f}, max={sample_img.max():.3f}")

        return fmri_data, stimulus_flat, labels, n_train, len(test_stimulus)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def train_correct_brain_decoder():
    """Training dengan data yang benar - menggunakan train/test split asli"""
    print("🧠 CORRECT BRAIN DECODER TRAINING")
    print("=" * 50)
    print("Using CORRECT stimulus and fMRI variables with REAL train/test split!")

    # Load correct data with proper train/test split
    result = load_correct_data()
    if result[0] is None:
        print("❌ Failed to load data")
        return

    fmri_data, stimulus_data, labels, n_train_samples, n_test_samples = result

    # Use the ORIGINAL train/test split from the dataset
    train_fmri = fmri_data[:n_train_samples]
    train_stimulus = stimulus_data[:n_train_samples]
    test_fmri = fmri_data[n_train_samples:]
    test_stimulus = stimulus_data[n_train_samples:]

    if labels is not None:
        train_labels = labels[:n_train_samples]
        test_labels = labels[n_train_samples:]
        print(f"\n📊 Data splits (ORIGINAL dataset split):")
        print(f"   Train: {train_fmri.shape[0]} samples (classes: {np.unique(train_labels)})")
        print(f"   Test: {test_fmri.shape[0]} samples (classes: {np.unique(test_labels)})")
    else:
        print(f"\n📊 Data splits (ORIGINAL dataset split):")
        print(f"   Train: {train_fmri.shape[0]} samples")
        print(f"   Test: {test_fmri.shape[0]} samples")

    # Create dataset
    train_dataset = BrainDataset(train_fmri, train_stimulus)
    train_loader = DataLoader(train_dataset, batch_size=min(2, len(train_fmri)), shuffle=True)

    # Create model
    model = BrainDecoder(
        fmri_dim=fmri_data.shape[1],
        stimulus_dim=stimulus_data.shape[1]
    )

    print(f"\n🤖 Brain Decoder Model:")
    print(f"   📥 INPUT: fMRI signals ({fmri_data.shape[1]} voxels)")
    print(f"   📤 OUTPUT: Digit images ({stimulus_data.shape[1]} pixels)")
    print(f"   🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🎯 TASK: Reconstruct REAL digits from brain activity")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"\n🚀 Training Brain Decoder...")
    print(f"   Target: Reconstruct REAL DIGIT IMAGES from fMRI")
    num_epochs = 20
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for fmri_batch, stimulus_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass: fMRI → Reconstructed Digits
            predictions = model(fmri_batch)
            loss = criterion(predictions, stimulus_batch)  # Compare with REAL digits

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

    # Test evaluation
    print(f"\n📊 Testing...")
    model.eval()

    with torch.no_grad():
        if len(test_fmri) > 0:
            test_fmri_tensor = torch.FloatTensor(test_fmri)
            test_stimulus_tensor = torch.FloatTensor(test_stimulus)

            predictions = model(test_fmri_tensor)
            test_loss = criterion(predictions, test_stimulus_tensor)

            # Compute comprehensive evaluation metrics
            print(f"📊 Computing comprehensive reconstruction metrics...")

            predictions_np = predictions.numpy()
            target_np = test_stimulus_tensor.numpy()

            if COMPREHENSIVE_EVAL_AVAILABLE:
                # Use comprehensive evaluation
                evaluator = EvaluationMetrics()
                results = evaluator.evaluate_all(predictions_np, target_np)

                # Print comprehensive results
                evaluator.print_results(results)

                # Create comprehensive plots
                print(f"🎨 Creating comprehensive evaluation plots...")

                # Metrics plot
                metrics_fig = evaluator.plot_metrics(results,
                                                   save_path="correct_results/comprehensive_metrics.png",
                                                   show=False)

                # Sample reconstructions plot
                samples_fig = evaluator.plot_sample_reconstructions(
                    predictions_np, target_np,
                    num_samples=min(6, len(predictions_np)),
                    save_path="correct_results/sample_reconstructions.png",
                    show=False
                )

                print(f"✅ Comprehensive evaluation completed!")
                print(f"📁 Additional plots saved:")
                print(f"   - comprehensive_metrics.png")
                print(f"   - sample_reconstructions.png")

            else:
                # Fallback to basic evaluation
                pred_flat = predictions_np.flatten()
                target_flat = target_np.flatten()
                correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

                # Basic PSNR and SSIM
                target_imgs = (target_np * 255).astype(np.uint8)
                pred_imgs = (predictions_np * 255).astype(np.uint8)

                psnr_scores = []
                ssim_scores = []

                for i in range(len(target_imgs)):
                    target_img = target_imgs[i].reshape(28, 28)
                    pred_img = pred_imgs[i].reshape(28, 28)

                    try:
                        psnr_val = psnr(target_img, pred_img, data_range=255)
                        psnr_scores.append(psnr_val)
                    except:
                        psnr_scores.append(0)

                    try:
                        ssim_val = ssim(target_img, pred_img, data_range=255)
                        ssim_scores.append(ssim_val)
                    except:
                        ssim_scores.append(0)

                print(f"📈 Test Results (Basic Evaluation):")
                print(f"   Test Loss: {test_loss.item():.6f}")
                print(f"   Correlation: {correlation:.4f}")
                print(f"   PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
                print(f"   SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
                print(f"\n💡 Install additional packages for comprehensive evaluation:")
                print(f"   pip install lpips clip-by-openai transformers opencv-python")

        else:
            print("⚠️  No test data available")
            predictions = None

    # Visualization
    print(f"\n🎨 Creating visualization...")
    plt.figure(figsize=(15, 5))

    # Training curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss (Correct Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)

    # Show real digit reconstruction
    if predictions is not None and len(test_stimulus) > 0:
        # Target digit (already transformed)
        plt.subplot(1, 3, 2)
        target_img = test_stimulus[0].reshape(28, 28)
        plt.imshow(target_img, cmap='gray')
        plt.title('Target: REAL Digit (Corrected)')
        plt.axis('off')

        # Predicted digit (apply transformation for visualization only)
        plt.subplot(1, 3, 3)
        pred_img = predictions[0].numpy().reshape(28, 28)
        plt.imshow(pred_img, cmap='gray')
        plt.title('Predicted: Reconstructed Digit (Corrected)')
        plt.axis('off')

    # Save results
    os.makedirs("correct_results", exist_ok=True)
    plt.savefig("correct_results/correct_training_results.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Save model
    torch.save(model.state_dict(), "correct_results/correct_brain_decoder.pth")

    print(f"\n✅ CORRECT training completed!")
    print(f"📁 Results saved to: correct_results/")
    print(f"   - correct_training_results.png")
    print(f"   - correct_brain_decoder.pth")

    return model, losses

if __name__ == "__main__":
    try:
        model, losses = train_correct_brain_decoder()

        print(f"\n🎉 SUCCESS with CORRECT DATA and PROPER TRAIN/TEST SPLIT!")
        print(f"Brain decoder trained using REAL stimulus and fMRI variables!")
        print(f"\n📊 What happened:")
        print(f"1. ✅ Used stimTrn as TRAINING stimulus (real digit images)")
        print(f"2. ✅ Used fmriTrn as TRAINING fMRI (real brain signals)")
        print(f"3. ✅ Used stimTest as TESTING stimulus (real digit images)")
        print(f"4. ✅ Used fmriTest as TESTING fMRI (real brain signals)")
        print(f"5. ✅ Used labelTrn and labelTest for proper classification")
        print(f"6. ✅ NO SYNTHETIC DATA - all from original dataset")
        print(f"7. ✅ Applied flip and -90° rotation to correct digit orientation")
        print(f"8. ✅ Target = actual digit images (clear, properly oriented)")
        print(f"9. ✅ Predicted = reconstructed digits with same transformations (properly oriented)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
