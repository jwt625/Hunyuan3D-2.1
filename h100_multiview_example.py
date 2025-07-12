#!/usr/bin/env python3
"""
H100 80GB Multi-View Hunyuan3D-2.1 Usage Example
================================================

This script demonstrates multi-view 3D generation using Hunyuan3D-2mv model,
optimized for NVIDIA H100 80GB. Similar to Gaussian splatting, this approach
uses multiple images of the same object from different viewpoints.

MULTI-VIEW CAPABILITIES:
========================
✅ Supports 1-4 input images from different viewpoints
✅ Front, Back, Left, Right view inputs
✅ Better reconstruction quality than single image
✅ More accurate geometry and texture details
✅ Similar workflow to Gaussian splatting but generates mesh

RECOMMENDED USAGE EXAMPLES:
==========================

1. SINGLE VIEW (Standard):
   python h100_multiview_example.py --front image.jpg

2. TWO VIEWS (Front + Back):
   python h100_multiview_example.py --front front.jpg --back back.jpg

3. FOUR VIEWS (Complete coverage):
   python h100_multiview_example.py --front front.jpg --back back.jpg --left left.jpg --right right.jpg

4. CUSTOM VIEWS (Any combination):
   python h100_multiview_example.py --front front.jpg --left left.jpg

MULTI-VIEW TIPS:
===============
- Use consistent lighting across all views
- Maintain similar object scale in all images
- Ensure object is centered in each view
- Remove backgrounds for best results (automatic)
- 90-degree rotations work best (front/back, left/right)

QUALITY COMPARISON:
==================
Single View:     Good quality, some guesswork on hidden areas
Multi-View (2):  Better geometry, improved texture consistency  
Multi-View (4):  Excellent quality, accurate 360° reconstruction
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


def setup_h100_optimizations():
    """Apply H100 specific optimizations."""
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(0.98)
    
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("✅ Enabled Flash Attention")
    except:
        pass
    
    print("🚀 Applied H100 80GB performance optimizations")


def check_gpu():
    """Check GPU and provide recommendations."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / (1024**3)
        
        print(f"🖥️  GPU: {props.name}")
        print(f"💾 Total VRAM: {memory_gb:.1f}GB")
        
        if "H100" in props.name and memory_gb > 70:
            print("🎉 H100 80GB detected - Perfect for multi-view generation!")
            return True
        elif memory_gb > 20:
            print("✅ High VRAM GPU - Multi-view should work well")
            return True
        else:
            print("⚠️  Limited VRAM - Consider single view generation")
            return False
    return False


def load_and_process_images(front=None, back=None, left=None, right=None, remove_bg=True):
    """Load and process multi-view images."""
    images = {}
    view_names = {'front': front, 'back': back, 'left': left, 'right': right}
    
    # Remove background processor
    rembg = BackgroundRemover() if remove_bg else None
    
    for view_name, image_path in view_names.items():
        if image_path and os.path.exists(image_path):
            print(f"📷 Loading {view_name} view: {image_path}")
            image = Image.open(image_path).convert("RGBA")
            
            # Remove background if needed
            if remove_bg and image.mode == 'RGB':
                print(f"🎭 Removing background from {view_name} view...")
                image = rembg(image)
            
            images[view_name] = image
    
    if not images:
        raise ValueError("No valid images provided!")
    
    print(f"✅ Loaded {len(images)} view(s): {list(images.keys())}")
    return images


def generate_multiview_shape(images, model_path='tencent/Hunyuan3D-2mv', output_path="output/multiview_mesh.glb"):
    """Generate 3D shape from multi-view images."""
    print("🏗️  Initializing multi-view shape generation pipeline...")
    
    # Clear cache before loading model
    torch.cuda.empty_cache()
    
    # Load the multi-view model
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    
    print(f"🎯 Generating 3D shape from {len(images)} view(s)...")
    start_time = time.time()
    
    # Generate mesh using multi-view input
    mesh = pipeline(image=images)[0]
    
    generation_time = time.time() - start_time
    print(f"⏱️  Multi-view shape generation completed in {generation_time:.1f}s")
    
    # Save the mesh
    print(f"💾 Saving multi-view mesh to: {output_path}")
    mesh.export(output_path)
    
    # Clear cache after generation
    torch.cuda.empty_cache()
    
    return mesh, output_path


def generate_texture_h100(mesh_path, images, output_path="output/multiview_textured.glb", 
                         max_views=8, resolution=768):
    """Generate texture for multi-view mesh."""
    print("🎨 Initializing texture generation for multi-view mesh...")
    
    # Clear cache before texture generation
    torch.cuda.empty_cache()
    
    # Configure with H100 optimized settings
    conf = Hunyuan3DPaintConfig(max_views, resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    
    # H100 optimizations for multi-view
    conf.render_size = 2048
    conf.texture_size = 4096
    conf.merge_method = "fast"
    
    print(f"📊 Texture settings: {max_views} views, {resolution}px resolution")
    print(f"🔧 Render: {conf.render_size}px, Texture: {conf.texture_size}px")
    
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    print("🖌️  Generating texture...")
    start_time = time.time()
    
    # Use the front view as reference for texture generation
    reference_image = images.get('front', list(images.values())[0])
    
    # Save reference image temporarily
    temp_ref_path = "temp_reference.png"
    reference_image.save(temp_ref_path)
    
    try:
        output_mesh_path = paint_pipeline(
            mesh_path=mesh_path,
            image_path=temp_ref_path,
            output_mesh_path=output_path
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
    
    generation_time = time.time() - start_time
    print(f"⏱️  Texture generation completed in {generation_time:.1f}s")
    
    # Clear cache after generation
    torch.cuda.empty_cache()
    
    return output_mesh_path


def main():
    parser = argparse.ArgumentParser(description="H100 Multi-View Hunyuan3D-2.1 Generation")
    parser.add_argument("--front", type=str, help="Front view image path")
    parser.add_argument("--back", type=str, help="Back view image path")
    parser.add_argument("--left", type=str, help="Left view image path")
    parser.add_argument("--right", type=str, help="Right view image path")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                       help="Output directory")
    parser.add_argument("--shape-only", action="store_true", 
                       help="Generate shape only (skip texture)")
    parser.add_argument("--no-background-removal", action="store_true", 
                       help="Skip automatic background removal")
    parser.add_argument("--max-views", type=int, default=8,
                       help="Maximum views for texture generation")
    parser.add_argument("--resolution", type=int, default=768,
                       help="Texture resolution")
    
    args = parser.parse_args()
    
    try:
        # Check if at least one image is provided
        if not any([args.front, args.back, args.left, args.right]):
            print("❌ Error: Please provide at least one view image")
            print("Example: python h100_multiview_example.py --front image.jpg")
            return
        
        # Check GPU and apply optimizations
        gpu_ok = check_gpu()
        if gpu_ok:
            setup_h100_optimizations()
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        # Load and process images
        images = load_and_process_images(
            front=args.front, back=args.back, left=args.left, right=args.right,
            remove_bg=not args.no_background_removal
        )
        
        # Generate multi-view shape
        shape_output = output_dir / "multiview_untextured.glb"
        mesh, mesh_path = generate_multiview_shape(images, output_path=str(shape_output))
        print(f"✅ Multi-view shape generation completed!")
        
        # Generate texture (unless shape-only mode)
        if not args.shape_only:
            textured_output = output_dir / "multiview_textured.glb"
            textured_mesh_path = generate_texture_h100(
                mesh_path, images, str(textured_output), 
                args.max_views, args.resolution
            )
            print(f"✅ Texture generation completed!")
            print(f"🎉 Final multi-view textured mesh: {textured_mesh_path}")
        else:
            print(f"🎉 Multi-view untextured mesh: {mesh_path}")
        
        # Memory usage summary
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (memory_used / memory_total) * 100
            print(f"📊 Peak GPU memory usage: {memory_used:.1f}GB / {memory_total:.1f}GB ({usage_percent:.1f}%)")
        
        print(f"\n🎊 Multi-view generation completed successfully!")
        print(f"📂 Check '{args.output_dir}' for your 3D models")
        print(f"🔍 Quality: Multi-view reconstruction with {len(images)} input view(s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
