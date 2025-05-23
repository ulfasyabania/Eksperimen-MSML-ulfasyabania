# syntax=docker/dockerfile:1
FROM continuumio/miniconda3
WORKDIR /app
COPY . /app
RUN conda env create -f conda.yaml
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "mlflow-env", "python", "modelling.py", "--input_csv", "bostonhousing_preprocessing.csv"]
