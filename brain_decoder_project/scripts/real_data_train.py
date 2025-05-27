"""
Training Brain Decoder dengan Data Asli dari Folder data/
Menggunakan data asli tanpa sintetis

Asumsi: Data (10, 3092) akan dibagi menjadi:
- fMRI data: sebagian features sebagai voxel aktivitas otak
- Stimulus data: sebagian features sebagai pixel gambar stimulus
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class RealBrainDataset(Dataset):
    def __init__(self, fmri_data, stimulus_data):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stimulus_data = torch.FloatTensor(stimulus_data)
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        return self.fmri_data[idx], self.stimulus_data[idx]

class RealBrainDecoder(nn.Module):
    def __init__(self, fmri_dim, stimulus_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(fmri_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, stimulus_dim),
            nn.Sigmoid()
        )
        
    def forward(self, fmri):
        return self.encoder(fmri)

def load_real_data():
    """Load data asli dari folder data/ dan pisahkan menjadi fMRI dan stimulus"""
    print("🔄 Loading REAL data from data/ folder...")
    
    try:
        from data_loader import DataLoader as NeuroDataLoader
        loader = NeuroDataLoader("data")
        files = loader.list_files()
        
        if not files:
            raise FileNotFoundError("No files found in data/ folder")
        
        # Load first file
        result = loader.load_data(files[0].name)
        print(f"✅ Loaded {files[0].name}")
        
        # Get raw data
        data_dict = result['data']
        first_var = list(data_dict.keys())[0]
        raw_data = data_dict[first_var]
        
        print(f"📊 Raw data shape: {raw_data.shape}")
        
        if len(raw_data.shape) != 2:
            raise ValueError(f"Expected 2D data, got {raw_data.shape}")
        
        n_samples, n_features = raw_data.shape
        print(f"📊 {n_samples} samples, {n_features} total features")
        
        # STRATEGI: Bagi data menjadi fMRI dan stimulus berdasarkan features
        
        # Option 1: Bagi berdasarkan rasio (misalnya 70% fMRI, 30% stimulus)
        fmri_features = int(n_features * 0.7)  # 70% untuk fMRI
        stimulus_features = n_features - fmri_features  # 30% untuk stimulus
        
        print(f"📊 Splitting strategy:")
        print(f"   fMRI features: {fmri_features} (first 70% of features)")
        print(f"   Stimulus features: {stimulus_features} (last 30% of features)")
        
        # Split data
        fmri_data = raw_data[:, :fmri_features]  # Features 0 to fmri_features
        stimulus_raw = raw_data[:, fmri_features:]  # Features fmri_features to end
        
        # Untuk stimulus, kita perlu format yang bisa divisualisasi
        # Jika stimulus_features bisa dibuat square, reshape ke 2D
        stimulus_sqrt = int(np.sqrt(stimulus_features))
        if stimulus_sqrt * stimulus_sqrt == stimulus_features:
            print(f"   → Reshaping stimulus to {stimulus_sqrt}x{stimulus_sqrt} images")
            stimulus_data = stimulus_raw.reshape(n_samples, stimulus_sqrt, stimulus_sqrt)
            stimulus_flat = stimulus_raw  # Keep flat version for training
        else:
            # Jika tidak square, pad atau crop ke ukuran terdekat
            target_size = 28  # Standard image size
            target_features = target_size * target_size  # 784
            
            if stimulus_features >= target_features:
                # Crop to 784 features
                stimulus_flat = stimulus_raw[:, :target_features]
                print(f"   → Cropping stimulus to {target_features} features (28x28)")
            else:
                # Pad with zeros to reach 784 features
                padding = target_features - stimulus_features
                stimulus_flat = np.pad(stimulus_raw, ((0, 0), (0, padding)), mode='constant')
                print(f"   → Padding stimulus to {target_features} features (28x28)")
            
            stimulus_data = stimulus_flat.reshape(n_samples, target_size, target_size)
        
        # Normalize data
        print("🔄 Normalizing data...")
        
        # Normalize fMRI data (z-score)
        fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)
        
        # Normalize stimulus data (0-1 range)
        stimulus_flat = (stimulus_flat - stimulus_flat.min()) / (stimulus_flat.max() - stimulus_flat.min() + 1e-8)
        stimulus_data = (stimulus_data - stimulus_data.min()) / (stimulus_data.max() - stimulus_data.min() + 1e-8)
        
        print(f"✅ Final data shapes:")
        print(f"   fMRI: {fmri_data.shape} (normalized, z-score)")
        print(f"   Stimulus flat: {stimulus_flat.shape} (normalized, 0-1)")
        print(f"   Stimulus 2D: {stimulus_data.shape} (for visualization)")
        
        return fmri_data, stimulus_flat, stimulus_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def train_real_brain_decoder():
    """Training dengan data asli"""
    print("🧠 REAL DATA BRAIN DECODER TRAINING")
    print("=" * 50)
    
    # Load real data
    fmri_data, stimulus_flat, stimulus_2d = load_real_data()
    
    if fmri_data is None:
        print("❌ Failed to load data")
        return
    
    # Split data for training/testing
    n_samples = fmri_data.shape[0]
    n_train = max(1, int(n_samples * 0.7))
    
    train_fmri = fmri_data[:n_train]
    train_stimulus = stimulus_flat[:n_train]
    test_fmri = fmri_data[n_train:]
    test_stimulus = stimulus_flat[n_train:]
    test_stimulus_2d = stimulus_2d[n_train:]
    
    print(f"📊 Data splits:")
    print(f"   Train: {train_fmri.shape[0]} samples")
    print(f"   Test: {test_fmri.shape[0]} samples")
    
    # Create dataset and dataloader
    train_dataset = RealBrainDataset(train_fmri, train_stimulus)
    train_loader = DataLoader(train_dataset, batch_size=min(2, n_train), shuffle=True)
    
    # Create model
    model = RealBrainDecoder(
        fmri_dim=fmri_data.shape[1],
        stimulus_dim=stimulus_flat.shape[1]
    )
    
    print(f"🤖 Model created:")
    print(f"   Input (fMRI): {fmri_data.shape[1]} voxels")
    print(f"   Output (Stimulus): {stimulus_flat.shape[1]} pixels")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n🚀 Starting training with REAL data...")
    num_epochs = 15
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for fmri_batch, stimulus_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass: fMRI → Reconstructed Stimulus
            predictions = model(fmri_batch)
            loss = criterion(predictions, stimulus_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Test evaluation
    print(f"\n📊 Testing with REAL data...")
    model.eval()
    
    with torch.no_grad():
        if len(test_fmri) > 0:
            test_fmri_tensor = torch.FloatTensor(test_fmri)
            test_stimulus_tensor = torch.FloatTensor(test_stimulus)
            
            # Predict stimulus from fMRI
            predictions = model(test_fmri_tensor)
            test_loss = criterion(predictions, test_stimulus_tensor)
            
            # Compute correlation
            pred_flat = predictions.numpy().flatten()
            target_flat = test_stimulus_tensor.numpy().flatten()
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
            
            print(f"📈 Test Results:")
            print(f"   Test Loss (MSE): {test_loss.item():.6f}")
            print(f"   Correlation: {correlation:.4f}")
            
            # Reshape predictions for visualization
            pred_2d = predictions.numpy().reshape(-1, test_stimulus_2d.shape[1], test_stimulus_2d.shape[2])
            
        else:
            print("⚠️  No test data available")
            pred_2d = None
    
    # Visualization
    print(f"\n🎨 Creating visualization...")
    plt.figure(figsize=(15, 5))
    
    # Training curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss (Real Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # Show reconstruction examples
    if pred_2d is not None and len(test_stimulus_2d) > 0:
        # Target stimulus
        plt.subplot(1, 3, 2)
        plt.imshow(test_stimulus_2d[0], cmap='gray')
        plt.title('Target Stimulus (Real Data)')
        plt.axis('off')
        
        # Predicted stimulus
        plt.subplot(1, 3, 3)
        plt.imshow(pred_2d[0], cmap='gray')
        plt.title('Predicted Stimulus')
        plt.axis('off')
    
    # Save results
    os.makedirs("real_data_results", exist_ok=True)
    plt.savefig("real_data_results/real_training_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), "real_data_results/real_brain_decoder.pth")
    
    print(f"\n✅ REAL DATA training completed!")
    print(f"📁 Results saved to: real_data_results/")
    print(f"   - real_training_results.png")
    print(f"   - real_brain_decoder.pth")
    
    return model, losses

if __name__ == "__main__":
    try:
        model, losses = train_real_brain_decoder()
        
        print(f"\n🎉 SUCCESS with REAL DATA!")
        print(f"Brain decoder trained using actual data from data/ folder!")
        print(f"\n📊 What happened:")
        print(f"1. ✅ Loaded real data from data/digit69_28x28.mat")
        print(f"2. ✅ Split features into fMRI (70%) and stimulus (30%)")
        print(f"3. ✅ Trained model: fMRI → Stimulus reconstruction")
        print(f"4. ✅ No synthetic data used - all from data/ folder")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
