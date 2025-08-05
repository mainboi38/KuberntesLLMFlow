FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create model cache directory
RUN mkdir -p /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV MODEL_NAME=meta-llama/Llama-2-7b-chat-hf

EXPOSE 8000

CMD ["python3", "app.py"]