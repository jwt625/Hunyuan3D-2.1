#!/usr/bin/env python3
"""
Test API server with images from the images folder
"""

import requests
import base64
import json
import time
import os
from PIL import Image
import io

API_BASE_URL = "http://localhost:8081"

def load_image_as_base64(image_path):
    """Load image and convert to base64"""
    image = Image.open(image_path).convert("RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_api_with_image(image_path):
    """Test API with a single image"""
    print(f"\nTesting API with: {image_path}")

    try:
        image_b64 = load_image_as_base64(image_path)

        request_data = {
            "image": image_b64,
            "type": "glb"
        }

        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/generate", json=request_data)

        if response.status_code == 200:
            duration = time.time() - start_time
            print(f"  ✓ Success in {duration:.1f}s")

            # Save result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            filename = f"api_output_{base_name}.glb"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"  Saved: {filename}")

        else:
            print(f"  ✗ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"  ✗ Exception: {e}")

def main():
    print("API Server Test with Images")
    print("=" * 40)

    # Test health first
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✓ API server is healthy")
        else:
            print("✗ API server health check failed")
            return
    except:
        print("✗ Cannot connect to API server")
        return

    # Test with images in images folder
    image_folder = "images"
    if os.path.exists(image_folder):
        for filename in sorted(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                test_api_with_image(image_path)
    else:
        print("Images folder not found")

if __name__ == "__main__":
    main()
