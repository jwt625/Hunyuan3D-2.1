# Optimized for H100 GPUs with CUDA 12.8 support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="hunyuan3d21-h100" maintainer="hunyuan3d21" \
      description="Hunyuan3D-2.1 optimized for H100 GPUs" \
      version="2.1-h100"

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Combine apt-get operations for better layer caching and smaller image size
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    vim \
    unzip \
    git-lfs \
    pkg-config \
    cmake \
    curl \
    # OpenGL and rendering libraries
    libegl1-mesa-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils-extra \
    libxrender1 \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Compilation dependencies
    libeigen3-dev \
    python3-dev \
    python3-setuptools \
    libcgal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment variables optimized for H100
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# CUDA environment variables optimized for H100 (compute capability 9.0)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# H100 optimized CUDA architectures (focus on 9.0 for H100, keep others for compatibility)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# H100 specific optimizations
ENV CUDA_NVCC_FLAGS="-allow-unsupported-compiler --use_fast_math"
ENV NCCL_DEBUG=WARN
ENV NCCL_SOCKET_IFNAME=^docker0

# Install conda with optimized settings
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# Initialize conda and create environment in one step for better caching
RUN conda init bash && \
    conda config --set always_yes true && \
    conda create -n hunyuan3d21 python=3.10 && \
    echo "source activate hunyuan3d21" > ~/.bashrc

ENV PATH="/workspace/miniconda3/envs/hunyuan3d21/bin:${PATH}"

# Install conda packages in one layer
RUN conda install -n hunyuan3d21 ninja -y && \
    conda install -n hunyuan3d21 cuda -c nvidia/label/cuda-12.4.1 -y && \
    conda install -n hunyuan3d21 libstdcxx-ng -c conda-forge -y

# Install PyTorch optimized for H100 with CUDA 12.4
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

# Clone Hunyuan3D-2.1 repository (use your fork for optimizations)
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git

# Set working directory to the cloned repository
WORKDIR /workspace/Hunyuan3D-2.1

# Install Python dependencies with optimizations
RUN pip install -r requirements.txt --no-cache-dir

# Install custom_rasterizer with H100 optimizations
RUN cd hy3dpaint/custom_rasterizer && \
    # H100 optimized compilation
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" && \
    export CUDA_NVCC_FLAGS="-allow-unsupported-compiler --use_fast_math" && \
    export MAX_JOBS=8 && \
    pip install -e . --no-cache-dir

# Install DifferentiableRenderer with optimizations
RUN cd hy3dpaint/DifferentiableRenderer && \
    export MAX_JOBS=8 && \
    bash compile_mesh_painter.sh

# Download required models and create directories
RUN cd hy3dpaint && \
    mkdir -p ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt && \
    mkdir -p ../gradio_cache

# Apply necessary path fixes
RUN cd hy3dpaint && \
    sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' textureGenPipeline.py && \
    cd utils && \
    sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' multiview_utils.py

# Set optimized library paths for H100
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"

# Activate conda environment by default
RUN echo "conda activate hunyuan3d21" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Expose ports for API server and Gradio
EXPOSE 7860 8081

# H100 specific runtime optimizations
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set default command to API server (optimized for production)
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "7860", "--limit-model-concurrency", "2"]
