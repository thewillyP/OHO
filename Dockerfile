FROM mambaorg/micromamba as dev

USER root

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server apt-utils bash build-essential ca-certificates curl git

RUN mkdir /var/run/sshd

# ssh server
EXPOSE 2222

# jupyter server
EXPOSE 9999

WORKDIR /workspace

COPY ./environment.yml .

COPY ./src ./src

RUN micromamba env create -f environment.yml && micromamba clean -afy


# FROM dev as prod 

