# Dockerfile for MetroSafetyAPI backend
# Use official Python slim image for minimal size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any). For torch, using CPU version.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app ./app

# Expose port (default 8000)
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV HOST=0.0.0.0 PORT=8000

# Command to run the server
CMD ["uvicorn", "app.main:app", "--host", "${HOST}", "--port", "${PORT}"]
