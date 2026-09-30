FROM python:3.11-slim

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user ID 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies first for caching
COPY --chown=user:user backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy all project files
COPY --chown=user:user . /app

# Ensure directories and permissions
RUN mkdir -p /app/backend/uploads && \
    chown -R user:user /app && \
    chmod -R 777 /app/backend/uploads

USER user

# Hugging Face Spaces port
EXPOSE 7860

# Start OralGuard AI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
