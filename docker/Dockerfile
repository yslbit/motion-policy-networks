# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION, University of Washington.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


FROM nvcr.io/nvidia/isaac-sim:2022.1.0

WORKDIR /root

# There is an issue with the Isaac Sim docker where it can't install from Apt
RUN echo "deb http://archive.ubuntu.com/ubuntu/ bionic main restricted" > /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic universe" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates universe" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic multiverse" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates multiverse" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security main restricted" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security universe" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security multiverse" >> /etc/apt/sources.list \
  && rm ~/.pip/pip.conf

# Section 1. Data Generation Tools
# Install apt dependencies necessary for OMPL
RUN apt update \
  && apt install -y --no-install-recommends \
    g++ \
    cmake \
    pkg-config \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    libode-dev \
    wget \
    libyaml-cpp-dev \
    python3.7 \
    python3.7-dev \
    libpython3.7-dev \
    libboost-python-dev \
    libboost-numpy-dev \
    git \
    curl \
    ninja-build \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
#   && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 \
#   && update-alternatives --set python3 /usr/bin/python3.7 \
#   && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
#   && python3.7 get-pip.py
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 \
  && curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
  && python3.7 get-pip.py

# Clone repos necessary to build OMPL and build it against the Python 3.7
RUN python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus numpy \
 && wget --tries=5 --waitretry=3 --read-timeout=30 --timeout=30 \
   -O /tmp/castxml.tar.gz \
   https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download \
 && tar zxf /tmp/castxml.tar.gz -C ${HOME} \
 && wget --tries=5 --waitretry=3 --read-timeout=30 --timeout=30 \
   -O /tmp/ompl-1.5.2.tar.gz \
   https://github.com/ompl/ompl/archive/1.5.2.tar.gz \
 && tar zxf /tmp/ompl-1.5.2.tar.gz \
 && rm -f /tmp/castxml.tar.gz /tmp/ompl-1.5.2.tar.gz \
 && export CXX=g++ \
 && export MAKEFLAGS="-j `nproc`" \
 && export PATH=${HOME}/castxml/bin:${PATH} \
 && mkdir -p ompl-1.5.2/build/Release \
 && cd ompl-1.5.2/build/Release \
 && cmake ../.. -DPYTHON_EXEC=/usr/bin/python3.7 \
 && make update_bindings \
 && make \
 && make install

# Set the PYTHONPATH so that Python has access to Lula (and Geometric Fabrics) and OMPL
# It appears that the apt installable version of python3.7 uses a Debian-only path, so
# adding the standard site-packages back into the PYTHONPATH (because it appears to be
# the default installation path for OMPL)
ENV PYTHONPATH=/isaac-sim/exts/omni.isaac.lula/pip_prebundle/:/usr/lib/python3.7/site-packages

# Local third-party packages bundled with this repo
COPY mpinets/third_party/geometrout /root/third_party/geometrout
COPY mpinets/third_party/robofin /root/third_party/robofin
COPY mpinets/third_party/atob-aubo /root/third_party/atob-aubo

# Section 2. Install Learning Tools (Cuda, Python Dependencies)
# Install cuda 11.3 (this should be upgraded if Pytorch is also upgraded)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
  && dpkg -i cuda-keyring_1.0-1_all.deb \
  && apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-3=11.3.109-1 \
    cuda-compat-11-3 \
    cuda-libraries-11-3=11.3.1-1 \
    libnpp-11-3=11.3.3.95-1 \
    cuda-nvtx-11-3=11.3.109-1 \
    libcusparse-11-3=11.6.0.109-1 \
    libcublas-11-3=11.5.1.109-1 \
    libnccl2=2.9.9-1+cuda11.3 \
    cuda-cudart-dev-11-3=11.3.109-1 \
    cuda-command-line-tools-11-3=11.3.1-1 \
    cuda-minimal-build-11-3=11.3.1-1 \
    cuda-libraries-dev-11-3=11.3.1-1 \
    cuda-nvml-dev-11-3=11.3.58-1 \
    cuda-nvprof-11-3=11.3.111-1 \
    libnpp-dev-11-3=11.3.3.95-1 \
    libcusparse-dev-11-3=11.6.0.109-1 \
    libcublas-dev-11-3=11.5.1.109-1 \
    libnccl-dev=2.9.9-1+cuda11.3 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ARG LD_LIBRARY_PATH=
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-11-3 libnccl2 libcublas-dev-11-3 libnccl-dev


# Install Python dependencies for Motion Policy Networks
# For some reason, the ikfast dependency do not get properly installed if
# ikfast and robofin are in the same pip install statement
RUN set -eux; \
  export PIP_DEFAULT_TIMEOUT=120; \
  retry_cmd() { \
    attempts=0; \
    max_attempts=3; \
    while true; do \
      if "$@"; then \
        break; \
      fi; \
      attempts=$((attempts + 1)); \
      if [ "$attempts" -ge "$max_attempts" ]; then \
        return 1; \
      fi; \
      sleep 10; \
    done; \
  }; \
  retry_cmd python3 -m pip install --upgrade pip setuptools; \
  retry_cmd python3 -m pip install \
    'torch<2' --extra-index-url https://download.pytorch.org/whl/cu113 \
    'pytorch_lightning<2' \
    h5py \
    trimesh \
    wandb \
    ipython \
    pybullet \
    pyquaternion \
    ikfast-pybind \
    numba \
    tqdm \
    urchin \
    yourdfpy \
    shapely \
    rtree \
    triangle \
    termcolor \
    meshcat; \
  retry_cmd python3 -m pip install \
    git+https://github.com/fishbotics/pointnet2_ops.git@v3.2.0; \
  python3 -m pip install --no-build-isolation --no-deps -e /root/third_party/geometrout; \
  python3 -m pip install --no-build-isolation --no-deps -e /root/third_party/robofin; \
  python3 -m pip install --no-build-isolation --no-deps -e /root/third_party/atob-aubo

# Section 3. Install ROS dependencies for the interactive tutorial
# Disabled by default because ROS Melodic is EOL and this large apt layer is
# both slow and brittle. Enable with `--build-arg INSTALL_ROS=1` when needed.
ARG INSTALL_ROS=0
RUN if [ "$INSTALL_ROS" = "1" ]; then \
    set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    APT_FETCH_OPTS="-o Acquire::Retries=10 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=120 -o Acquire::https::Timeout=120"; \
    retry_apt_install() { \
      attempts=0; \
      max_attempts=3; \
      while true; do \
        if apt-get ${APT_FETCH_OPTS} install "$@"; then \
          break; \
        fi; \
        attempts=$((attempts + 1)); \
        if [ "$attempts" -ge "$max_attempts" ]; then \
          return 1; \
        fi; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 15; \
        apt-get ${APT_FETCH_OPTS} update; \
      done; \
    }; \
    retry_cmd() { \
      attempts=0; \
      max_attempts=3; \
      while true; do \
        if "$@"; then \
          break; \
        fi; \
        attempts=$((attempts + 1)); \
        if [ "$attempts" -ge "$max_attempts" ]; then \
          return 1; \
        fi; \
        sleep 5; \
      done; \
    }; \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && echo "America/Los_Angeles" > /etc/timezone \
    && apt-get ${APT_FETCH_OPTS} update \
    && retry_apt_install -y --no-install-recommends \
      curl \
      dirmngr \
      gnupg \
      ca-certificates \
      apt-transport-https \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sh -c 'echo "deb http://snapshots.ros.org/melodic/2023-06-27/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list' \
    && retry_cmd apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA \
    && apt-get ${APT_FETCH_OPTS} update \
    && retry_apt_install -y --no-install-recommends --fix-missing \
      ros-melodic-desktop-full \
      python-rosdep \
      python-rosinstall \
      python-rosinstall-generator \
      python-wstool \
      ros-melodic-catkin \
      python-catkin-tools \
      ros-melodic-moveit \
      ros-melodic-moveit-python \
      ros-melodic-franka-ros \
      ros-melodic-tf2-ros \
      python3-rospkg-modules \
      python3-catkin-pkg-modules \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc \
    && if ! retry_cmd rosdep init && ! test -f /etc/ros/rosdep/sources.list.d/20-default.list; then \
         echo "Warning: rosdep init timed out during image build; continuing without preinitialized rosdep sources."; \
       fi \
    && if ! retry_cmd rosdep update --include-eol-distros; then \
         echo "Warning: rosdep update timed out during image build; you can rerun it later inside the container."; \
       fi \
    && python3 -m pip install rospkg netifaces ; \
  fi

ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311

# Section 4. Final setup items
ENTRYPOINT []
