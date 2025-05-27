"""
View Architecture Diagrams

This script opens the generated architecture diagrams for review.

Author: AI Assistant
Date: 2024
"""

import os
import subprocess
import sys
from pathlib import Path

def open_file(filepath):
    """Open file with default system application."""
    try:
        if sys.platform.startswith('win'):
            os.startfile(filepath)
        elif sys.platform.startswith('darwin'):  # macOS
            subprocess.run(['open', filepath])
        else:  # Linux
            subprocess.run(['xdg-open', filepath])
        return True
    except Exception as e:
        print(f"❌ Could not open {filepath}: {e}")
        return False

def main():
    """Open all architecture diagrams."""
    
    print("🎨 ARCHITECTURE DIAGRAMS VIEWER")
    print("=" * 50)
    
    diagrams_dir = Path("architecture_diagrams")
    
    if not diagrams_dir.exists():
        print("❌ Architecture diagrams not found!")
        print("💡 Run: python create_architecture_diagram.py")
        return
    
    # List of diagrams to open
    diagrams = [
        ("brain_decoder_architecture.png", "Main Brain Decoder Architecture"),
        ("detailed_network_architecture.png", "Detailed Network Architecture"),
        ("data_flow_diagram.png", "Data Flow and Processing Pipeline"),
        ("transformer_architecture_detail.png", "Transformer Architecture Details"),
        ("evaluation_metrics_diagram.png", "Comprehensive Evaluation Metrics")
    ]
    
    print("📊 Available diagrams:")
    for i, (filename, description) in enumerate(diagrams, 1):
        filepath = diagrams_dir / filename
        if filepath.exists():
            print(f"   {i}. {description}")
            print(f"      File: {filename}")
        else:
            print(f"   {i}. {description} - ❌ NOT FOUND")
    
    print("\n🚀 Opening diagrams...")
    
    opened_count = 0
    for filename, description in diagrams:
        filepath = diagrams_dir / filename
        if filepath.exists():
            print(f"📖 Opening: {description}")
            if open_file(str(filepath)):
                opened_count += 1
            else:
                print(f"⚠️  Could not open: {filename}")
        else:
            print(f"⚠️  File not found: {filename}")
    
    print(f"\n✅ Opened {opened_count}/{len(diagrams)} diagrams")
    
    if opened_count > 0:
        print("\n📝 Diagram Information:")
        print("• All diagrams are 300 DPI publication quality")
        print("• PNG files for presentations and review")
        print("• PDF files for journal submissions")
        print("• Designed for high-impact journal standards")
        
        print("\n🎯 Usage Guidelines:")
        print("• Main Architecture: Use as primary figure in paper")
        print("• Detailed Network: Use in methods section")
        print("• Data Flow: Use in supplementary material")
        print("• Transformer Details: Use in technical appendix")
        print("• Evaluation Metrics: Use in results section")
        
        print("\n📚 For detailed documentation, see:")
        print("• ARCHITECTURE_DIAGRAMS_README.md")
    
    else:
        print("\n💡 To generate diagrams, run:")
        print("python create_architecture_diagram.py")

if __name__ == "__main__":
    main()
