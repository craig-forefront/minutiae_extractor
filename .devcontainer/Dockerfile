FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install all packages in one step
RUN dnf install -y \
    dnf-plugins-core \
    git \
    wget \
    tar \
    python3.11 \
    python3.11-devel \
    vim \
    sudo \
    gcc \
    gcc-c++ \
    # OpenCV dependencies
    mesa-libGL \
    libgomp \
    # Install Node.js
    nodejs \
    npm \
    # Image processing libraries
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    # Scientific computing
    openblas-devel \
    lapack-devel \
    && dnf clean all

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create non-root user
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install CUDA toolkit
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y cuda-toolkit-12-8 && \
    dnf clean all

# Install cuDNN for TensorFlow
RUN dnf install -y libcudnn9-cuda-12 libcudnn9-devel-cuda-12 && \
    dnf clean all

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Switch to vscode user
USER vscode
WORKDIR /workspace

# Install uv for vscode user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.local/bin:${PATH}"

# Add workspace venv to PATH (created by uv sync)
ENV PATH="/workspace/.venv/bin:${PATH}"
