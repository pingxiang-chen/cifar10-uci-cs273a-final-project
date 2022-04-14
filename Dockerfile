FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        vim \
        ffmpeg \ 
        libsm6 \ 
        libxext6 \
        python3-tk \
        python3-setuptools \
        systemd \
        wget \
        iproute2 \
        iputils-ping \
        software-properties-common

RUN pip3 install --upgrade pip 
RUN pip3 install matplotlib
WORKDIR /source