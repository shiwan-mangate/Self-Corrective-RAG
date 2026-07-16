# =========================================================================
# Step 1: Use an official lightweight Python runtime
# =========================================================================
FROM python:3.11-slim

# =========================================================================
# Step 2: Set strict performance & architecture environment variables
# =========================================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/model_cache \
    HF_HOME=/app/model_cache

# Set the execution working directory
WORKDIR /app

# =========================================================================
# Step 3: Install minimal system layer dependencies
# =========================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
# Step 4: Leverage Docker layer caching for Python dependencies
# =========================================================================
COPY requirements.txt .

# Force pip to download the ultra-lightweight CPU-only wheel for PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# =========================================================================
# Step 5: Bake the Embedding Model into the Image Layer (Crucial for Render)
# =========================================================================
# This runs during build time so the model is already present when booting.
# CHANGE 'all-MiniLM-L6-v2' to your exact sentence-transformer model identifier.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# =========================================================================
# Step 6: Copy application source code
# =========================================================================
COPY . .

# Expose the internal container communication port
EXPOSE 10000

# =========================================================================
# Step 7: Define the production startup command
# =========================================================================
# Wrapped in a shell invocation context to dynamically bind to Render's $PORT.
# Includes proxy header flags to maintain semantic integrity for Swagger UI (/docs).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*'"]