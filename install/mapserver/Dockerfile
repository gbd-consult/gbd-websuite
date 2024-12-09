FROM ubuntu:24.04
LABEL Description="MapServer build image"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update \
&& apt -y upgrade \
&& apt -y install software-properties-common \
&& apt -y update \
&& apt -y install \
    build-essential \
    cmake \
    protobuf-c-compiler \
    libprotobuf-c-dev \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libproj-dev \
    pkg-config \
    libfribidi-dev \
    libharfbuzz-dev \
    libcairo2-dev \
    libfcgi-dev \
    libgeos-dev \
    libpq-dev \
    libgdal-dev \
    libxml2-dev \
    swig \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv
