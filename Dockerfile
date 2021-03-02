##JSS
FROM tensorflow/tensorflow:1.14.0-gpu-py3


# Prepare and empty machine for building
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# Build and install ceres solver
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j4 && \
    make install

RUN apt-get install -y cuda-libraries-10-0
RUN apt-get install -y cuda-libraries-dev-10-0
#RUN apt install cuda-libraries-10-0

# Build and install COLMAP
# Note: This Dockerfile has been tested using COLMAP pre-release 3.6.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
WORKDIR /
RUN git clone https://github.com/colmap/colmap.git #--branch 3.6
RUN cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install

RUN apt-get update -y
RUN apt-get install python3 python3-pip unzip wget -y
WORKDIR /home

RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab notebook
RUN pip3 install git+https://github.com/mihaidusmanu/pycolmap
RUN git clone -b combination https://github.com/kronecker08/Hierarchical-Localization.git
RUN cd Hierarchical-Localization && git submodule update --init --recursive
WORKDIR /home/Hierarchical-Localization
RUN pip3 install -r requirements.txt
RUN bash weight.sh


