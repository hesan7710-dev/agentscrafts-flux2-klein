FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/runpod-volume/huggingface
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY worker.py prompts.json ./
CMD ["python", "-u", "worker.py"]
