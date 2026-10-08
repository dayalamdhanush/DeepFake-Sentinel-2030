# Dockerfile - DeepFake Sentinel 2030 Backend
FROM python:3.12-slim

# Install system dependencies (ffmpeg for video/audio, libsndfile for librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend and ML source files
COPY backend /app/backend
COPY ml /app/ml
COPY scripts /app/scripts
COPY models /app/models
COPY datasets /app/datasets

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
