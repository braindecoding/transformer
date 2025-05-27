"""
Install Dependencies for Comprehensive Evaluation

This script installs the required packages for comprehensive brain decoder evaluation.

Author: AI Assistant
Date: 2024
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"🔄 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🧠 BRAIN DECODER EVALUATION DEPENDENCIES INSTALLER")
    print("=" * 60)
    
    # List of required packages
    packages = [
        "scikit-image>=0.18.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0"
    ]
    
    # Optional advanced packages (may require special installation)
    advanced_packages = [
        "lpips",
        "transformers>=4.20.0"
    ]
    
    print("📦 Installing basic packages...")
    basic_success = True
    for package in packages:
        if not install_package(package):
            basic_success = False
    
    print("\n📦 Installing advanced packages...")
    print("⚠️  Note: Some packages may require additional dependencies")
    
    advanced_success = True
    for package in advanced_packages:
        if not install_package(package):
            advanced_success = False
    
    # Special handling for CLIP
    print("\n📦 Installing CLIP...")
    clip_success = install_package("git+https://github.com/openai/CLIP.git")
    if not clip_success:
        print("⚠️  Trying alternative CLIP installation...")
        clip_success = install_package("clip-by-openai")
    
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    if basic_success:
        print("✅ Basic evaluation packages installed successfully!")
    else:
        print("❌ Some basic packages failed to install")
    
    if advanced_success:
        print("✅ Advanced evaluation packages installed successfully!")
    else:
        print("⚠️  Some advanced packages failed to install")
    
    if clip_success:
        print("✅ CLIP package installed successfully!")
    else:
        print("⚠️  CLIP package installation failed")
    
    print("\n🎯 What's available now:")
    print("   ✅ MSE, PSNR, SSIM metrics")
    print("   ✅ FID calculation")
    print("   ✅ Comprehensive plotting")
    
    if advanced_success:
        print("   ✅ LPIPS perceptual similarity")
    else:
        print("   ⚠️  LPIPS perceptual similarity (fallback mode)")
    
    if clip_success:
        print("   ✅ CLIP semantic similarity")
    else:
        print("   ⚠️  CLIP semantic similarity (fallback mode)")
    
    print("\n🚀 You can now run comprehensive evaluation!")
    print("   python src/brain_decoder_main.py")
    print("   python scripts/comprehensive_evaluation.py")

if __name__ == "__main__":
    main()
