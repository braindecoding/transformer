"""
Check data structure untuk memahami apa sebenarnya isi digit69_28x28.mat
"""

import numpy as np
from data_loader import DataLoader

def check_data_structure():
    print("🔍 CHECKING DATA STRUCTURE")
    print("=" * 40)
    
    loader = DataLoader("data")
    result = loader.load_data("digit69_28x28.mat")
    
    print(f"📁 File: digit69_28x28.mat")
    print(f"Variables: {list(result['data'].keys())}")
    
    for var_name, var_data in result['data'].items():
        print(f"\n📊 Variable: {var_name}")
        print(f"   Shape: {var_data.shape}")
        print(f"   Type: {var_data.dtype}")
        print(f"   Min: {var_data.min():.3f}")
        print(f"   Max: {var_data.max():.3f}")
        print(f"   Mean: {var_data.mean():.3f}")
        
        # Analisis berdasarkan nama file dan shape
        if "28x28" in "digit69_28x28.mat":
            print(f"   → File name suggests 28x28 digit images")
            
        if var_data.shape[1] == 784:  # 28*28 = 784
            print(f"   → 784 features = 28*28 pixels → LIKELY DIGIT IMAGES")
            # Show first image
            first_image = var_data[0].reshape(28, 28)
            print(f"   → First image stats: min={first_image.min():.3f}, max={first_image.max():.3f}")
            
        elif var_data.shape[1] > 1000:
            print(f"   → Many features ({var_data.shape[1]}) → Could be fMRI voxels")
            
        # Show some sample values
        print(f"   First 5 values of sample 0: {var_data[0, :5]}")

if __name__ == "__main__":
    check_data_structure()
