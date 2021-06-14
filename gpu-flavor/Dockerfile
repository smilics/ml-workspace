ARG ARG_WORKSPACE_BASE_IMAGE="mltooling/ml-workspace:latest"
# Build from full flavor of workspace with same version
FROM $ARG_WORKSPACE_BASE_IMAGE

ARG ARG_WORKSPACE_FLAVOR="gpu"
ENV WORKSPACE_FLAVOR=$ARG_WORKSPACE_FLAVOR

USER root

### NVIDIA CUDA BASE ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu18.04-x86_64/base/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.0.3

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-0=11.0.221-1 \
    cuda-compat-11-0 \
    && ln -s cuda-11.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"



### CUDA RUNTIME ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu18.04-x86_64/runtime/Dockerfile

ENV NCCL_VERSION 2.9.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-0=11.0.3-1 \
    libnpp-11-0=11.1.0.245-1 \
    cuda-nvtx-11-0=11.0.167-1 \
    libcublas-11-0=11.2.0.252-1 \
    libcusparse-11-0=11.1.1.245-1 \
    libnccl2=$NCCL_VERSION-1+cuda11.0 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-11-0 libnccl2



### END CUDA RUNTIME ###

### CUDA DEVEL ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu18.04-x86_64/devel/Dockerfile

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-11-0=11.0.221-1 \
    cuda-command-line-tools-11-0=11.0.3-1 \
    cuda-minimal-build-11-0=11.0.3-1 \
    cuda-libraries-dev-11-0=11.0.3-1 \
    cuda-nvml-dev-11-0=11.0.167-1 \
    libnpp-dev-11-0=11.1.0.245-1 \
    libnccl-dev=2.9.6-1+cuda11.0 \
    libcublas-dev-11-0=11.2.0.252-1 \
    libcusparse-dev-11-0=11.1.1.245-1 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-dev-11-0 libnccl-dev
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs



### END CUDA DEVEL ###

### CUDANN8 DEVEL ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu18.04-x86_64/devel/cudnn7/Dockerfile

ENV CUDNN_VERSION 8.0.5.39

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.0 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.0 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

### END CUDANN8 ###

# Link Cupti:
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64

# # Install TensorRT. Requires that libcudnn7 is installed above.
# # https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer7=7.1.3-1+cuda11.0 \
        libnvinfer-dev=7.1.3-1+cuda11.0 \
        libnvinfer-plugin7=7.1.3-1+cuda11.0 && \

    # Cleanup
    clean-layer.sh

### GPU DATA SCIENCE LIBRARIES ###

RUN \
    apt-get update && \
    apt-get install -y libomp-dev libopenblas-base && \
    # Not needed? Install cuda-toolkit (e.g. for pytorch: https://pytorch.org/): https://anaconda.org/anaconda/cudatoolkit
    conda install -y cudatoolkit=11.0 -c pytorch && \
    # Install cupy: https://cupy.chainer.org/
    pip install --no-cache-dir cupy-cuda110 && \
    # Install pycuda: https://pypi.org/project/pycuda
    pip install --no-cache-dir pycuda && \
    # Install gpu utils libs
    pip install --no-cache-dir gpustat py3nvml gputil && \
    # Install scikit-cuda: https://scikit-cuda.readthedocs.io/en/latest/install.html
    pip install --no-cache-dir scikit-cuda && \
    # Install tensorflow gpu
    pip uninstall -y tensorflow tensorflow-cpu intel-tensorflow && \
    # TODO: tensorflow 2.3.1 installs tenorboard 2.4.0 with problems, use 2.3.0
    pip install --no-cache-dir tensorflow-gpu==2.4.0 && \
    # Install ONNX GPU Runtime
    # TODO: 1.4.x is latest with cuda 10.1 support
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu==1.4.0 && \
    # Install pytorch gpu
    # uninstall cpu only packages via conda
    conda remove --force -y pytorch cpuonly && \
    # https://pytorch.org/get-started/locally/
    conda install -y pytorch -c pytorch && \
    # Install faiss gpu
    conda remove --force -y faiss-cpu && \
    conda install -y faiss-gpu -c pytorch && \
    # Update mxnet to gpu edition
    pip uninstall -y mxnet-mkl && \
    pip install --no-cache-dir mxnet-cu110 && \
    # install jax: https://github.com/google/jax#pip-installation
    pip install --upgrade jax jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html  && \
    # Install pygpu - Required for theano: http://deeplearning.net/software/libgpuarray/
    conda install -y pygpu && \
    # Install lightgbm
    pip uninstall -y lightgbm && \
    pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"  && \
    # nvidia python ml lib
    pip install --upgrade --force-reinstall nvidia-ml-py3 && \
    # SpeedTorch: https://github.com/Santosh-Gupta/SpeedTorch
    pip install --no-cache-dir SpeedTorch && \
    # Ipyexperiments - fix memory leaks
    pip install --no-cache-dir ipyexperiments && \
    # Cleanup
    clean-layer.sh

# TODO: nvdashboard does not work with relative paths
# RUN \
#     # Install Jupyterlab GPU Plugin: https://github.com/rapidsai/jupyterlab-nvdashboard
#     pip install jupyterlab-nvdashboard && \
#     jupyter labextension install jupyterlab-nvdashboard && \
#     # Clean jupyter lab cache: https://github.com/jupyterlab/jupyterlab/issues/4930
#     jupyter lab clean && \
#     jlpm cache clean && \
#     # Remove build folder -> should be remove by lab clean as well?
#     rm -rf $CONDA_ROOT/share/jupyter/lab/staging && \
#     # Cleanup
#     clean-layer.sh

# TODO install DALI: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#dali-and-ngc
# TODO: if > Ubuntu 19.04 -> install nvtop: https://github.com/Syllo/nvtop
# TODO: Install Arrrayfire: https://arrayfire.com/download/ pip install --no-cache-dir arrayfire && \
# TODO Nvidia Apex: https://github.com/NVIDIA/apex

# cd $RESOURCES_PATH && \
# git clone https://github.com/NVIDIA/apex && \
# cd apex  && \
# # Surpress output - if there is a problem remove to see logs &> /dev/null
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
# rm -rf apex && \

# https://www.anaconda.com/getting-started-with-gpu-computing-in-anaconda/

# By default, the majority of GPU memory will be allocated by the first
# execution of a TensorFlow graph. While this behavior can be desirable for
# production pipelines, it is less desirable for interactive use. Set
# TF_FORCE_GPU_ALLOW_GROWTH to change this default behavior as if the user had
ENV TF_FORCE_GPU_ALLOW_GROWTH true

### END DATA SCIENCE LIBRARIES ###

### GPU TOOLS ###

### END GPU TOOLS ###

### CONFIGURATION ###

#TODO: tests are currently empty COPY resources/tests/ /resources/tests

# argument needs to be initalized again
ARG ARG_WORKSPACE_VERSION="latest"
ENV WORKSPACE_VERSION=$ARG_WORKSPACE_VERSION

# Overwrite & add Labels
ARG ARG_BUILD_DATE="unknown"
ARG ARG_VCS_REF="unknown"

LABEL \
    "workspace.version"=$WORKSPACE_VERSION \
    "workspace.flavor"=$WORKSPACE_FLAVOR \
    "workspace.baseimage"=$ARG_WORKSPACE_BASE_IMAGE \
    "org.opencontainers.image.version"=$WORKSPACE_VERSION \
    "org.opencontainers.image.revision"=$ARG_VCS_REF \
    "org.opencontainers.image.created"=$ARG_BUILD_DATE \
    "org.label-schema.version"=$WORKSPACE_VERSION \
    "org.label-schema.vcs-ref"=$ARG_VCS_REF \
    "org.label-schema.build-date"=$ARG_BUILD_DATE

