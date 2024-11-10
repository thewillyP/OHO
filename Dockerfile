FROM mambaorg/micromamba:cuda12.4.1-ubuntu22.04

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
    graphviz \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]
# Optional: Set bash as the default for login shells
RUN echo "exec /bin/bash" >> /etc/profile


WORKDIR /workspace

COPY ./environment.yml .

COPY ./src ./src

RUN micromamba create -f environment.yml && micromamba clean --all --yes