FROM continuumio/miniconda3

WORKDIR /workspace

COPY ./environment.yml .

COPY ./src ./src

RUN conda env create -f environment.yml


