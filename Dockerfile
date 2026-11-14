FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/runpod-volume/huggingface
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY worker.py prompts.json ./
CMD ["python", "-u", "worker.py"]
