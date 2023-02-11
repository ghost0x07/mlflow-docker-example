FROM python:3.10

RUN pip install pandas==1.5.3 && \
    pip install mlflow==2.1.1 && \
    pip install numba==0.56.4

WORKDIR /app
COPY . .