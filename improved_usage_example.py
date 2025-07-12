#!/usr/bin/env python3
"""
Improved Hunyuan3D-2.1 Usage Example
====================================

This script demonstrates how to use Hunyuan3D-2.1 for both shape generation and texture painting,
with proper file saving and error handling.

Requirements:
- Activate the virtual environment: source venv/bin/activate
- Ensure you have an input image (PNG/JPG format)
- Make sure you have sufficient GPU memory (10GB+ for shape, 21GB+ for texture)
"""

import sys
import os
from pathlib import Path
from PIL import Image
import argparse

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
except Exception as e:
    print(f"⚠️  Warning: Failed to apply torchvision fix: {e}")


def setup_output_directory(output_dir: str = "output"):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path


def preprocess_image(image_path: str, remove_background: bool = True):
    """Load and preprocess the input image."""
    print(f"📷 Loading image from: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGBA")
    
    # Remove background if the image is RGB (no transparency)
    if image.mode == 'RGB' and remove_background:
        print("🎭 Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    
    return image


def generate_shape(image, model_path: str = 'tencent/Hunyuan3D-2.1', output_path: str = "output/shape.glb"):
    """Generate 3D shape from image."""
    print("🏗️  Initializing shape generation pipeline...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    
    print("🎯 Generating 3D shape...")
    mesh = pipeline_shapegen(image=image)[0]
    
    # Save the untextured mesh
    print(f"💾 Saving untextured mesh to: {output_path}")
    mesh.export(output_path)
    
    return mesh, output_path


def generate_texture(mesh_path: str, image_path: str, output_path: str = "output/textured_shape.glb",
                    max_num_view: int = 6, resolution: int = 512, enable_memory_optimization: bool = True):
    """Generate texture for the 3D mesh with memory optimization for RTX 4090."""
    print("🎨 Initializing texture generation pipeline...")

    # Configure the paint pipeline with RTX 4090 optimizations
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    # RTX 4090 Memory Optimizations
    if enable_memory_optimization:
        # Reduce render and texture sizes for 24GB VRAM
        conf.render_size = 1024  # Reduced from 2048
        conf.texture_size = 2048  # Reduced from 4096
        conf.merge_method = "fast"  # Use fast merge method
        print("🔧 Applied RTX 4090 memory optimizations")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    print(f"🖌️  Generating texture with {max_num_view} views at {resolution}x{resolution} resolution...")
    print(f"📊 Render size: {conf.render_size}, Texture size: {conf.texture_size}")

    # Clear GPU cache before texture generation
    import torch
    torch.cuda.empty_cache()

    output_mesh_path = paint_pipeline(
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=output_path
    )

    # Clear cache after generation
    torch.cuda.empty_cache()

    print(f"💾 Saving textured mesh to: {output_mesh_path}")
    return output_mesh_path


def main():
    parser = argparse.ArgumentParser(description="Generate 3D models using Hunyuan3D-2.1")
    parser.add_argument("--image", "-i", type=str, default="assets/demo.png", 
                       help="Path to input image (default: assets/demo.png)")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                       help="Output directory (default: output)")
    parser.add_argument("--shape-only", action="store_true", 
                       help="Generate shape only (skip texture generation)")
    parser.add_argument("--no-background-removal", action="store_true", 
                       help="Skip background removal")
    parser.add_argument("--max-views", type=int, default=6, choices=[6, 7, 8, 9],
                       help="Maximum number of views for texture generation (default: 6, RTX 4090 optimized)")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 768],
                       help="Texture resolution (default: 512, RTX 4090 optimized)")
    parser.add_argument("--model-path", type=str, default="tencent/Hunyuan3D-2.1",
                       help="Model path or HuggingFace model ID")
    parser.add_argument("--disable-memory-optimization", action="store_true",
                       help="Disable RTX 4090 memory optimizations (may cause OOM)")
    parser.add_argument("--high-quality", action="store_true",
                       help="Use high quality settings (8 views, 768 resolution) - may require more VRAM")
    
    args = parser.parse_args()
    
    try:
        # Apply high-quality settings if requested
        if args.high_quality:
            args.max_views = 8
            args.resolution = 768
            print("🔥 High-quality mode enabled: 8 views, 768 resolution")
            print("⚠️  This may require more VRAM and take longer")

        # Setup
        output_dir = setup_output_directory(args.output_dir)
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"🖥️  Detected RTX 4090 with 24GB VRAM - optimizations enabled")

        # Preprocess image
        image = preprocess_image(args.image, remove_background=not args.no_background_removal)

        # Generate shape
        shape_output = output_dir / "untextured_mesh.glb"
        mesh, mesh_path = generate_shape(image, args.model_path, str(shape_output))
        print(f"✅ Shape generation completed!")

        # Generate texture (unless shape-only mode)
        if not args.shape_only:
            textured_output = output_dir / "textured_mesh.glb"
            textured_mesh_path = generate_texture(
                mesh_path, args.image, str(textured_output),
                args.max_views, args.resolution,
                enable_memory_optimization=not args.disable_memory_optimization
            )
            print(f"✅ Texture generation completed!")
            print(f"🎉 Final textured mesh saved to: {textured_mesh_path}")
        else:
            print(f"🎉 Untextured mesh saved to: {mesh_path}")

        print("\n🎊 Generation completed successfully!")
        print(f"📂 Check the '{args.output_dir}' directory for your 3D models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
