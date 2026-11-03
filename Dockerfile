FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the model named here is baked into the image; app.py loads MODEL_NAME at runtime.
ARG MODEL_NAME=turbo
ENV MODEL_NAME=${MODEL_NAME}

COPY preload_models.py .
RUN python3 /preload_models.py && rm /preload_models.py

CMD ["python3", "app.py"]
