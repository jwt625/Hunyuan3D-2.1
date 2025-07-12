#!/usr/bin/env python3
"""
System Compatibility Check for Hunyuan3D-2.1
=============================================

This script checks your system compatibility and recommends optimal settings.
"""

import sys
import torch
import platform
import psutil
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10 - Perfect!")
        return True
    elif version.major == 3 and version.minor in [8, 9, 11]:
        print("⚠️  Python version should work, but 3.10 is recommended")
        return True
    else:
        print("❌ Python version may not be compatible")
        return False

def check_gpu():
    """Check GPU compatibility and memory."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU required for Hunyuan3D-2.1")
        return False, None
    
    gpu_count = torch.cuda.device_count()
    print(f"🖥️  GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name}")
        print(f"   Memory: {memory_gb:.1f}GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        
        # Check if it's RTX 4090
        if "RTX 4090" in props.name:
            print("🚀 RTX 4090 detected - Excellent for Hunyuan3D-2.1!")
            return True, "rtx4090"
        elif memory_gb >= 20:
            print("✅ High VRAM GPU - Should work well")
            return True, "high_vram"
        elif memory_gb >= 15:
            print("⚠️  Medium VRAM - May need optimization")
            return True, "medium_vram"
        elif memory_gb >= 10:
            print("⚠️  Low VRAM - Shape generation only recommended")
            return True, "low_vram"
        else:
            print("❌ Insufficient VRAM for Hunyuan3D-2.1")
            return False, "insufficient"
    
    return False, None

def check_system_memory():
    """Check system RAM."""
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💾 System RAM: {memory_gb:.1f}GB total, {available_gb:.1f}GB available")
    
    if memory_gb >= 32:
        print("✅ Excellent RAM for Hunyuan3D-2.1")
        return True
    elif memory_gb >= 16:
        print("✅ Sufficient RAM")
        return True
    else:
        print("⚠️  Low RAM - may affect performance")
        return False

def check_pytorch():
    """Check PyTorch installation."""
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"⚡ CUDA Version: {cuda_version}")
        
        if cuda_version and cuda_version.startswith("12."):
            print("✅ CUDA 12.x - Perfect for Hunyuan3D-2.1")
            return True
        else:
            print("⚠️  CUDA version may not be optimal")
            return True
    else:
        print("❌ CUDA not available in PyTorch")
        return False

def check_dependencies():
    """Check if key dependencies are available."""
    dependencies = [
        ("diffusers", "Diffusion models"),
        ("transformers", "Transformer models"),
        ("accelerate", "Model acceleration"),
        ("trimesh", "3D mesh processing"),
        ("PIL", "Image processing")
    ]
    
    all_good = True
    print("📦 Checking Dependencies:")
    
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep} - {desc}")
        except ImportError:
            print(f"   ❌ {dep} - {desc} (Missing)")
            all_good = False
    
    return all_good

def recommend_settings(gpu_type):
    """Recommend optimal settings based on hardware."""
    print("\n🎛️  Recommended Settings:")
    
    if gpu_type == "rtx4090":
        print("   Profile: balanced or high_quality")
        print("   Max Views: 7-8")
        print("   Resolution: 512-768")
        print("   Expected Generation Time: 2-5 minutes")
        print("   Memory Usage: 18-23GB VRAM")
    elif gpu_type == "high_vram":
        print("   Profile: balanced")
        print("   Max Views: 6-7")
        print("   Resolution: 512")
        print("   Expected Generation Time: 3-7 minutes")
    elif gpu_type == "medium_vram":
        print("   Profile: efficient")
        print("   Max Views: 6")
        print("   Resolution: 512")
        print("   Expected Generation Time: 4-8 minutes")
    elif gpu_type == "low_vram":
        print("   Recommendation: Shape generation only")
        print("   Texture generation may cause OOM")
    else:
        print("   Hardware not suitable for Hunyuan3D-2.1")

def main():
    print("🔍 Hunyuan3D-2.1 System Compatibility Check")
    print("=" * 50)
    
    # Check all components
    python_ok = check_python_version()
    gpu_ok, gpu_type = check_gpu()
    memory_ok = check_system_memory()
    pytorch_ok = check_pytorch()
    deps_ok = check_dependencies()
    
    print("\n📊 Summary:")
    print("=" * 20)
    
    if all([python_ok, gpu_ok, memory_ok, pytorch_ok, deps_ok]):
        print("🎉 System is ready for Hunyuan3D-2.1!")
        recommend_settings(gpu_type)
        
        print("\n🚀 Quick Start Commands:")
        if gpu_type == "rtx4090":
            print("   python rtx4090_optimized_example.py --profile balanced")
            print("   python rtx4090_optimized_example.py --profile high_quality")
        else:
            print("   python improved_usage_example.py --max-views 6 --resolution 512")
            
    else:
        print("⚠️  Some issues detected. Please resolve them before using Hunyuan3D-2.1")
        
        if not gpu_ok:
            print("   - GPU issues detected")
        if not deps_ok:
            print("   - Missing dependencies (run: pip install -r requirements.txt)")
        if not pytorch_ok:
            print("   - PyTorch/CUDA issues")

if __name__ == "__main__":
    main()
