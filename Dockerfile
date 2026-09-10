# Define base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    ninja-build \
    python3-pip \
    build-essential \
    cmake \
    libdynamicedt3d-dev \
    openssh-server \
    acl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch
RUN pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create a non-root user
ARG USER_NAME
ARG USER_ID
RUN useradd -m --no-log-init --uid ${USER_ID} ${USER_NAME} -G sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set workspace path
WORKDIR /workspace/src/scripts

# Copy entrypoint script and make it executable
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Expose SSH port (if needed)
EXPOSE 22
EXPOSE 6006

# Set PYTHONPATH
ENV PYTHONPATH="/workspace/src:${PYTHONPATH:-}"

# Set entrypoint
ENTRYPOINT ["/home/${USER_NAME}/entrypoint.sh"]
