# The build-stage image:
FROM continuumio/miniconda3 AS build

# Install the package as normal:
COPY environment.yml .
RUN conda env create -f environment.yml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n master -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# When image is run, run the code with the environment
# activated:
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
    python -c "import numpy; print('success!')"


# FROM mambaorg/micromamba AS build

# COPY ./environment.yml .

# RUN micromamba env create -f environment.yml 

# # Use conda-pack to create a standalone enviornment
# # in /venv:
# RUN /bin/bash -c 'micromamba shell init -s bash && source ~/.bashrc && micromamba activate $(grep -m 1 \"name: \" environment.yml | cut -d \" \" -f 2) && conda-pack -n master -o /tmp/env.tar && \
#     mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#     rm /tmp/env.tar'

# # RUN conda-pack -n master -o /tmp/env.tar && \
# #     mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
# #     rm /tmp/env.tar

# # We've put venv in same path it'll be in final image,
# # so now fix up paths:
# RUN /venv/bin/conda-unpack

# # The runtime-stage image; we can use Debian as the
# # base image since the Conda env also includes Python
# # for us.
# FROM debian:bookworm-slim AS runtime

# USER root

# # Copy /venv from the previous stage:
# COPY --from=build /venv /venv

# ENV PATH="/venv/bin:$PATH"

# RUN apt update && apt install -y --no-install-recommends openssh-server apt-utils bash build-essential ca-certificates curl git && apt clean && rm -rf /var/lib/apt/lists/*

# RUN mkdir /var/run/sshd

# WORKDIR /workspace

# COPY ./src ./src

# RUN pip install -e ./src/code

# # When image is run, run the code with the environment
# # activated:
# SHELL ["/bin/bash", "-c"]
# ENTRYPOINT source /venv/bin/activate && \
#     python -c "import numpy; print('success!')"


# # ssh server
# EXPOSE 2222
# # jupyter server
# EXPOSE 9999



