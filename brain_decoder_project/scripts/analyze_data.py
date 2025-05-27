"""
Analyze data structure untuk memahami format data asli
"""

import numpy as np
from data_loader import DataLoader

def analyze_data():
    print("🔍 ANALYZING DATA STRUCTURE")
    print("=" * 40)
    
    loader = DataLoader("data")
    files = loader.list_files()
    
    print(f"📁 Files found: {len(files)}")
    for file in files:
        print(f"   - {file.name}")
    
    if files:
        # Load first file
        result = loader.load_data(files[0].name)
        print(f"\n📊 Loading: {files[0].name}")
        print(f"Variables: {list(result['data'].keys())}")
        
        for var_name, var_data in result['data'].items():
            if hasattr(var_data, 'shape'):
                print(f"\n🔍 Variable: {var_name}")
                print(f"   Shape: {var_data.shape}")
                print(f"   Type: {var_data.dtype}")
                print(f"   Min: {var_data.min():.3f}")
                print(f"   Max: {var_data.max():.3f}")
                print(f"   Mean: {var_data.mean():.3f}")
                
                # Analyze structure
                if len(var_data.shape) == 2:
                    n_samples, n_features = var_data.shape
                    print(f"   → {n_samples} samples, {n_features} features")
                    
                    # Check if it could be images
                    if n_features == 784:
                        print(f"   → Likely 28x28 images (784 = 28*28)")
                    elif n_features > 1000:
                        print(f"   → Likely fMRI data (many voxels)")
                    
                    # Show first few values
                    print(f"   First 5 values: {var_data[0, :5]}")
                    
                elif len(var_data.shape) == 3:
                    print(f"   → 3D data: {var_data.shape}")
                    if var_data.shape[1] == var_data.shape[2]:
                        print(f"   → Likely images: {var_data.shape[0]} images of {var_data.shape[1]}x{var_data.shape[2]}")

if __name__ == "__main__":
    analyze_data()
