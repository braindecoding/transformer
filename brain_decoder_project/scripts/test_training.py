"""
Test script untuk training brain decoder
"""

import numpy as np
import torch
print("🔄 Testing imports...")

try:
    from brain_decoder import BrainDecoder, BrainDecoderTrainer, BrainDecodingDataset
    print("✅ Brain decoder imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

try:
    from data_loader import DataLoader as NeuroDataLoader
    print("✅ Data loader import successful")
except Exception as e:
    print(f"❌ Data loader import error: {e}")
    exit(1)

print("\n🔄 Loading data...")
try:
    loader = NeuroDataLoader("data")
    files = loader.list_files()
    print(f"📁 Found {len(files)} files")
    
    if files:
        result = loader.load_data(files[0].name)
        print(f"✅ Loaded {files[0].name}")
        print(f"📊 Data variables: {list(result['data'].keys())}")
        
        # Get first variable
        first_var = list(result['data'].keys())[0]
        raw_data = result['data'][first_var]
        print(f"📊 Data shape: {raw_data.shape}")
        
        # Process data
        if len(raw_data.shape) == 2:
            n_samples, n_features = raw_data.shape
            print(f"📊 {n_samples} samples, {n_features} features")
            
            # Use data as fMRI
            fmri_data = raw_data.copy()
            
            # Create simple stimulus data
            stimulus_data = np.random.rand(n_samples, 28, 28)
            
            print(f"✅ fMRI shape: {fmri_data.shape}")
            print(f"✅ Stimulus shape: {stimulus_data.shape}")
            
            print("\n🤖 Creating model...")
            model = BrainDecoder(
                fmri_input_dim=fmri_data.shape[1],
                stimulus_shape=(28, 28),
                stimulus_channels=1,
                hidden_dim=32,  # Very small for test
                num_encoder_layers=1,
                num_decoder_layers=1,
                nhead=2,
                dropout=0.1
            )
            print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            print("\n🔄 Testing forward pass...")
            fmri_tensor = torch.FloatTensor(fmri_data).unsqueeze(1)  # Add time dim
            with torch.no_grad():
                output = model(fmri_tensor)
            print(f"✅ Forward pass successful: {output.shape}")
            
            print("\n🎉 All tests passed! Training should work.")
            
        else:
            print(f"❌ Unexpected data shape: {raw_data.shape}")
    else:
        print("❌ No files found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
