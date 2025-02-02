## Docker

## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06

ENV DEBIAN_FRONTEND=noninteractive

# Update and install tzdata separately
RUN apt update && apt install -y tzdata

# Install necessary packages
RUN apt install -y git && \
    apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \
    apt clean && apt install wget && rm -rf /var/lib/apt/lists/*

ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6"

COPY environment.yml /tmp/environment.yml
COPY gaussian_splatting/ /tmp/gaussian_splatting/
WORKDIR /tmp/
RUN conda env create --file environment.yml && conda init bash && exec bash && conda activate gaussian_splatting
RUN rm /tmp/environment.yml

# Install colmap
RUN apt update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

RUN git clone https://github.com/colmap/colmap.git
WORKDIR /tmp/colmap
# Back up commit: 98940342171e27fbf7a52223a39b5b3f699f23b8
RUN git checkout 682ea9ac4020a143047758739259b3ff04dabe8d &&\
    mkdir build && cd build &&\
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=all-major &&\
    ninja &&\
    ninja install


# Install Node.js 21.x at the system level
RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
    apt update && apt-get install -y \
    nodejs \
    aptitude

RUN aptitude install -y npm

# Ensure the system Node.js takes priority over Conda's Node.js
RUN echo 'export PATH=/usr/bin:$PATH' >> /etc/profile.d/system_node.sh && \
    chmod +x /etc/profile.d/system_node.sh

WORKDIR /sugar

RUN pip3 install google-api-python-client google-auth google-auth-oauthlib watchdog

# Default conda project
RUN echo "conda activate sugar" >> ~/.bashrc

# This error occurs because there’s a conflict between the threading layer used
# by Intel MKL (Math Kernel Library) and the libgomp library, 
# which is typically used by OpenMP for parallel processing. 
# This often happens when libraries like NumPy or SciPy are used in combination
# with a multithreaded application (e.g., your Docker container or Python environment).
# Solution, set threading layer explicitly! (GNU or INTEL)
ENV MKL_THREADING_LAYER=GNU

# Set up Meshroom paths
RUN echo "export ALICEVISION_ROOT=/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/" >> ~/.bashrc &&\
    echo "export PATH=$PATH:/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/bin/" >> ~/.bashrc &&\
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/lib/" >> ~/.bashrc &&\
    echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc
