#!/usr/bin/env python3
"""
H100 80GB Optimized Hunyuan3D-2.1 Usage Example
===============================================

This script is optimized for NVIDIA H100 80GB - the ultimate configuration
for Hunyuan3D-2.1 with maximum quality and performance.

H100 Advantages:
- 80GB HBM3 memory (3x more than RTX 4090)
- Superior compute performance
- Can run maximum quality settings
- Support for multiple concurrent generations

RECOMMENDED USAGE EXAMPLES:
==========================

1. QUICK START (Recommended for most users):
   python h100_optimized_example.py --image your_image.png --profile high_quality

   Output: High-quality 3D model with 9 views, 768px resolution
   Time: ~3-4 minutes, VRAM: ~35GB

2. MAXIMUM QUALITY (Best results):
   python h100_optimized_example.py --image your_image.png --profile maximum

   Output: Outstanding quality with 12 views, 768px resolution
   Time: ~4-6 minutes, VRAM: ~45GB

3. ULTRA QUALITY (Experimental - pushes H100 limits):
   python h100_optimized_example.py --image your_image.png --profile ultra

   Output: Ultimate quality with 12 views, 3072px render, 6144px texture
   Time: ~6-10 minutes, VRAM: ~60GB

4. SHAPE ONLY (Fast preview):
   python h100_optimized_example.py --image your_image.png --shape-only

   Output: 3D shape without texture
   Time: ~30-60 seconds, VRAM: ~10GB

5. BATCH PROCESSING (Multiple images):
   for img in *.png; do
       python h100_optimized_example.py --image "$img" --profile high_quality --output-dir "output_$img"
   done

QUALITY PROFILES COMPARISON:
===========================

Profile      | Views | Resolution | Render Size | Texture Size | Est. VRAM | Quality Level
-------------|-------|------------|-------------|--------------|-----------|---------------
standard     |   8   |    768px   |    2048px   |    4096px    |   ~25GB   | Good
high_quality |   9   |    768px   |    2048px   |    4096px    |   ~35GB   | Excellent ⭐
maximum      |  12   |    768px   |    2048px   |    4096px    |   ~45GB   | Outstanding
ultra        |  12   |    768px   |    3072px   |    6144px    |   ~60GB   | Ultimate

⭐ = Recommended starting point

PERFORMANCE EXPECTATIONS ON H100:
=================================
- Shape Generation: 30-60 seconds
- Texture Generation: 1-6 minutes (profile dependent)
- Total Generation Time: 2-10 minutes
- Memory Usage: 20-60GB VRAM (you have 80GB available!)
- Concurrent Runs: Can run 2-3 generations simultaneously

TROUBLESHOOTING:
===============
- Out of Memory: Use 'standard' profile or --shape-only
- Slow Performance: Check if other processes are using GPU
- Poor Quality: Try 'maximum' or 'ultra' profile
- File Not Found: Ensure image path is correct and accessible

ADVANCED OPTIONS:
================
--image PATH              Input image path (PNG, JPG, JPEG)
--output-dir DIR          Output directory (default: output)
--profile PROFILE         Quality profile (standard|high_quality|maximum|ultra)
--shape-only              Generate shape only, skip texture generation
--no-background-removal   Skip automatic background removal

EXAMPLES WITH ADVANCED OPTIONS:
==============================
# Custom output directory
python h100_optimized_example.py --image photo.jpg --output-dir results/photo1

# Skip background removal (if image already has transparent background)
python h100_optimized_example.py --image cutout.png --no-background-removal

# Generate only 3D shape for quick preview
python h100_optimized_example.py --image sketch.png --shape-only

# Maximum quality with custom output
python h100_optimized_example.py --image portrait.jpg --profile maximum --output-dir high_quality_results
"""

import sys
import os
import torch
from pathlib import Path
from PIL import Image
import argparse
import time

# Add the required paths
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# Apply torchvision fix if available
try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("✅ Applied torchvision compatibility fix")
except ImportError:
    print("⚠️  Warning: torchvision_fix module not found, proceeding without compatibility fix")


class H100Config:
    """H100 80GB optimized configuration profiles."""
    
    # Standard quality (conservative)
    STANDARD = {
        "max_views": 8,
        "resolution": 768,
        "render_size": 2048,
        "texture_size": 4096,
        "description": "Standard quality for H100"
    }
    
    # High quality (recommended)
    HIGH_QUALITY = {
        "max_views": 9,
        "resolution": 768,
        "render_size": 2048,
        "texture_size": 4096,
        "description": "High quality - recommended for H100"
    }
    
    # Maximum quality (uses H100's full potential)
    MAXIMUM = {
        "max_views": 12,
        "resolution": 768,
        "render_size": 2048,
        "texture_size": 4096,
        "description": "Maximum quality - full H100 potential"
    }
    
    # Ultra quality (experimental)
    ULTRA = {
        "max_views": 12,
        "resolution": 768,
        "render_size": 3072,
        "texture_size": 6144,
        "description": "Ultra quality - experimental settings"
    }


def setup_h100_optimizations():
    """Apply H100 specific optimizations."""
    # Enable all performance optimizations
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')  # H100 can handle high precision
    
    # Use more memory since H100 has 80GB
    torch.cuda.set_per_process_memory_fraction(0.98)
    
    # Enable optimized attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("✅ Enabled Flash Attention")
    except:
        pass
    
    print("🚀 Applied H100 80GB performance optimizations")


def check_h100_memory():
    """Check H100 memory and provide recommendations."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / (1024**3)
        
        print(f"🖥️  GPU: {props.name}")
        print(f"💾 Total VRAM: {memory_gb:.1f}GB")
        
        if "H100" in props.name and memory_gb > 70:
            print("🎉 H100 80GB detected - Ultimate Hunyuan3D-2.1 performance!")
            return "h100_80gb"
        elif memory_gb > 40:
            print("✅ High-end GPU detected")
            return "high_end"
        else:
            return "standard"
    return "unknown"


def generate_shape_h100(image, model_path: str = 'tencent/Hunyuan3D-2.1', output_path: str = "output/shape.glb"):
    """Generate 3D shape optimized for H100."""
    print("🏗️  Initializing shape generation pipeline (H100 optimized)...")
    
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    
    print("🎯 Generating 3D shape...")
    start_time = time.time()
    
    mesh = pipeline_shapegen(image=image)[0]
    
    generation_time = time.time() - start_time
    print(f"⏱️  Shape generation completed in {generation_time:.1f}s")
    
    # Save the untextured mesh
    print(f"💾 Saving untextured mesh to: {output_path}")
    mesh.export(output_path)
    
    return mesh, output_path


def generate_texture_h100(mesh_path: str, image_path: str, output_path: str, profile: dict):
    """Generate texture optimized for H100."""
    print(f"🎨 Initializing texture generation (H100 Profile: {profile['description']})...")
    
    # Configure with H100 optimized settings
    conf = Hunyuan3DPaintConfig(profile["max_views"], profile["resolution"])
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    
    # Apply H100 optimizations
    conf.render_size = profile["render_size"]
    conf.texture_size = profile["texture_size"]
    conf.merge_method = "fast"
    
    print(f"📊 Settings: {profile['max_views']} views, {profile['resolution']}px resolution")
    print(f"🔧 Render: {conf.render_size}px, Texture: {conf.texture_size}px")
    
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    print("🖌️  Generating texture...")
    start_time = time.time()
    
    output_mesh_path = paint_pipeline(
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=output_path
    )
    
    generation_time = time.time() - start_time
    print(f"⏱️  Texture generation completed in {generation_time:.1f}s")
    
    return output_mesh_path


def main():
    parser = argparse.ArgumentParser(description="H100 80GB Optimized Hunyuan3D-2.1 Generation")
    parser.add_argument("--image", "-i", type=str, default="assets/demo.png", 
                       help="Path to input image")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                       help="Output directory")
    parser.add_argument("--profile", type=str, default="high_quality", 
                       choices=["standard", "high_quality", "maximum", "ultra"],
                       help="Quality profile for H100 (default: high_quality)")
    parser.add_argument("--shape-only", action="store_true", 
                       help="Generate shape only (skip texture)")
    parser.add_argument("--no-background-removal", action="store_true", 
                       help="Skip background removal")
    
    args = parser.parse_args()
    
    try:
        # Check GPU and apply optimizations
        gpu_type = check_h100_memory()
        if gpu_type == "h100_80gb":
            setup_h100_optimizations()
        else:
            print(f"⚠️  Warning: Optimized for H100 80GB, detected {gpu_type}")
        
        # Select profile
        profile_map = {
            "standard": H100Config.STANDARD,
            "high_quality": H100Config.HIGH_QUALITY,
            "maximum": H100Config.MAXIMUM,
            "ultra": H100Config.ULTRA
        }
        profile = profile_map[args.profile]
        print(f"🎛️  Using {args.profile} profile: {profile['description']}")
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        # Load and preprocess image
        print(f"📷 Loading image: {args.image}")
        image = Image.open(args.image).convert("RGBA")
        
        if image.mode == 'RGB' and not args.no_background_removal:
            print("🎭 Removing background...")
            rembg = BackgroundRemover()
            image = rembg(image)
        
        # Generate shape
        shape_output = output_dir / "untextured_mesh.glb"
        mesh, mesh_path = generate_shape_h100(image, output_path=str(shape_output))
        print(f"✅ Shape generation completed!")
        
        # Generate texture (unless shape-only mode)
        if not args.shape_only:
            textured_output = output_dir / "textured_mesh.glb"
            textured_mesh_path = generate_texture_h100(
                mesh_path, args.image, str(textured_output), profile
            )
            print(f"✅ Texture generation completed!")
            print(f"🎉 Final textured mesh: {textured_mesh_path}")
        else:
            print(f"🎉 Untextured mesh: {mesh_path}")
        
        # Memory usage summary
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (memory_used / memory_total) * 100
            print(f"📊 Peak GPU memory usage: {memory_used:.1f}GB / {memory_total:.1f}GB ({usage_percent:.1f}%)")
        
        print(f"\n🎊 Generation completed successfully!")
        print(f"📂 Check '{args.output_dir}' for your 3D models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
