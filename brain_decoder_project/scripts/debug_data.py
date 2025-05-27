"""
Debug data untuk melihat apa sebenarnya isi digit69_28x28.mat
dan mencari stimulus yang benar
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader

def debug_data():
    print("🔍 DEBUGGING DATA STRUCTURE")
    print("=" * 50)
    
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
        print(f"   Std: {var_data.std():.3f}")
        
        # Coba berbagai cara untuk menemukan digit images
        if len(var_data.shape) == 2:
            n_samples, n_features = var_data.shape
            
            print(f"\n🔍 Analyzing {n_samples} samples with {n_features} features:")
            
            # Test 1: Coba 784 features pertama
            if n_features >= 784:
                first_784 = var_data[:, :784]
                print(f"\n📊 First 784 features (should be 28x28 digits):")
                print(f"   Min: {first_784.min():.3f}, Max: {first_784.max():.3f}")
                
                # Visualize first few samples
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                fig.suptitle('First 784 features reshaped to 28x28')
                
                for i in range(min(10, n_samples)):
                    row = i // 5
                    col = i % 5
                    
                    img = first_784[i].reshape(28, 28)
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].set_title(f'Sample {i}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig('debug_first_784.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            # Test 2: Coba 784 features terakhir
            if n_features >= 784:
                last_784 = var_data[:, -784:]
                print(f"\n📊 Last 784 features (might be 28x28 digits):")
                print(f"   Min: {last_784.min():.3f}, Max: {last_784.max():.3f}")
                
                # Visualize last 784 features
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                fig.suptitle('Last 784 features reshaped to 28x28')
                
                for i in range(min(10, n_samples)):
                    row = i // 5
                    col = i % 5
                    
                    img = last_784[i].reshape(28, 28)
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].set_title(f'Sample {i}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig('debug_last_784.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            # Test 3: Coba middle 784 features
            if n_features >= 784:
                start_idx = (n_features - 784) // 2
                middle_784 = var_data[:, start_idx:start_idx+784]
                print(f"\n📊 Middle 784 features (features {start_idx}-{start_idx+784}):")
                print(f"   Min: {middle_784.min():.3f}, Max: {middle_784.max():.3f}")
                
                # Visualize middle 784 features
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                fig.suptitle('Middle 784 features reshaped to 28x28')
                
                for i in range(min(10, n_samples)):
                    row = i // 5
                    col = i % 5
                    
                    img = middle_784[i].reshape(28, 28)
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].set_title(f'Sample {i}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig('debug_middle_784.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            # Test 4: Coba cari pola yang terlihat seperti digit
            print(f"\n🔍 Searching for digit-like patterns...")
            
            # Coba berbagai range 784 features
            best_range = None
            best_score = -1
            
            for start in range(0, n_features-784, 100):  # Check every 100 features
                test_data = var_data[:, start:start+784]
                
                # Score berdasarkan variance dan range
                variance = test_data.var()
                data_range = test_data.max() - test_data.min()
                score = variance * data_range
                
                if score > best_score:
                    best_score = score
                    best_range = (start, start+784)
            
            if best_range:
                start, end = best_range
                best_784 = var_data[:, start:end]
                print(f"   Best range found: features {start}-{end}")
                print(f"   Score: {best_score:.3f}")
                print(f"   Min: {best_784.min():.3f}, Max: {best_784.max():.3f}")
                
                # Visualize best range
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                fig.suptitle(f'Best 784 features (features {start}-{end})')
                
                for i in range(min(10, n_samples)):
                    row = i // 5
                    col = i % 5
                    
                    img = best_784[i].reshape(28, 28)
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].set_title(f'Sample {i}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig('debug_best_784.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                return best_784, var_data[:, :start] if start > 0 else var_data[:, end:]
    
    return None, None

def test_different_normalizations(data):
    """Test berbagai cara normalisasi"""
    print(f"\n🔧 Testing different normalizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    img = data[0].reshape(28, 28)
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    
    # Min-max normalization
    normalized = (data - data.min()) / (data.max() - data.min())
    img = normalized[0].reshape(28, 28)
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].set_title('Min-Max (0-1)')
    
    # Z-score normalization
    normalized = (data - data.mean()) / data.std()
    img = normalized[0].reshape(28, 28)
    axes[0, 2].imshow(img, cmap='gray')
    axes[0, 2].set_title('Z-score')
    
    # Clip and normalize
    clipped = np.clip(data, np.percentile(data, 5), np.percentile(data, 95))
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    img = normalized[0].reshape(28, 28)
    axes[0, 3].imshow(img, cmap='gray')
    axes[0, 3].set_title('Clipped + Normalized')
    
    # Test with different samples
    for i in range(4):
        if i < data.shape[0]:
            normalized = (data - data.min()) / (data.max() - data.min())
            img = normalized[i].reshape(28, 28)
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Sample {i} (Min-Max)')
        axes[1, i].axis('off')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_normalizations.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    stimulus_data, fmri_data = debug_data()
    
    if stimulus_data is not None:
        print(f"\n✅ Found potential stimulus data!")
        test_different_normalizations(stimulus_data)
        
        print(f"\n📊 Recommendations:")
        print(f"1. Use the 'best' 784 features as stimulus")
        print(f"2. Try different normalization methods")
        print(f"3. Check debug_*.png files for visual inspection")
    else:
        print(f"\n❌ Could not find clear digit patterns")
        print(f"Data might not contain recognizable digits")
