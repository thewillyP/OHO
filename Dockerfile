FROM mambaorg/micromamba

USER root

RUN apt update && apt install -y --no-install-recommends openssh-server apt-utils bash build-essential ca-certificates curl git && apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

# ssh server
EXPOSE 2222
# jupyter server
EXPOSE 9999

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]
# Optional: Set bash as the default for login shells
RUN echo "exec /bin/bash" >> /etc/profile

WORKDIR /workspace

COPY ./environment.yml .

COPY ./src ./src

RUN micromamba env create -f environment.yml && micromamba clean -afy