#!/usr/bin/env python3
"""
Main script to run brain decoder training
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main training
try:
    from brain_decoder_main import train_correct_brain_decoder
except ImportError as e:
    print(f"Error importing brain_decoder_main: {e}")
    print("Make sure you're running from the brain_decoder_project directory")
    sys.exit(1)

if __name__ == "__main__":
    print("BRAIN DECODER PROJECT")
    print("=" * 50)
    print("Starting brain decoder training...")
    
    try:
        model, losses = train_correct_brain_decoder()
        print("\nTraining completed successfully!")
        print("Check results/ folder for outputs")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
