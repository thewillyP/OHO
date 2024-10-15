FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Set up ssh server stuff
RUN mkdir /var/run/sshd

RUN apt update \
    && apt install -y --no-install-recommends \
    openssh-server \
    apt-utils \
    bash \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    nano \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]
# Optional: Set bash as the default for login shells
RUN echo "exec /bin/bash" >> /etc/profile


ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -u -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda clean -afy


WORKDIR /workspace

COPY ./environment.yml .

COPY ./src ./src

RUN conda env create -f environment.yml && conda clean -afy
