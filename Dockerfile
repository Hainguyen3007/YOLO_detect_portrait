# Stage 1: Builder - Install dependencies
FROM python:3.10-slim AS builder

# Combine RUN commands to reduce layers and clean up after installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies (CPU-only) and clean up cache
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run.py .
COPY 123_aug90_model_ft/weights/best.pt 123_aug90_model_ft/weights/best.pt

# Stage 2: Runtime - Lightweight image
FROM python:3.10-slim

# Combine RUN commands to reduce layers and clean up after installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser
USER appuser

# Set working directory
WORKDIR /app

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/run.py .
COPY --from=builder /app/123_aug90_model_ft/weights/best.pt 123_aug90_model_ft/weights/best.pt

# Expose port
EXPOSE 8000

# Use ENTRYPOINT for better flexibility and CMD for default arguments
ENTRYPOINT ["uvicorn"]
CMD ["run:app", "--host", "0.0.0.0", "--port", "8000"]
