FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 1. Pin numpy<2 first (torch 2.x compiled against numpy 1.x)
RUN pip install --no-cache-dir "numpy<2"

# 2. Install CPU-only PyTorch 2.6.0 (required by latest transformers >= CVE-2025-32434 fix)
RUN pip install --no-cache-dir \
    "torch==2.6.0+cpu" \
    "torchvision==0.21.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining deps, excluding torch/torchvision/transformers to control versions
COPY requirements.txt .
RUN grep -vE "^torch|^transformers" requirements.txt > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt

# Pin transformers to 4.47.1 — last stable version before 4.50 CLIP refactor
# (4.50+ changed get_text_features() return type breaking compatibility)
RUN pip install --no-cache-dir "transformers==4.47.1"

# 4. Pre-download CLIP model into image (avoids timeout on Cloud Run startup)
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
print('Downloading CLIP model...'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('Done')"

# Copy application source
COPY src/ ./src/
COPY run.py .

RUN mkdir -p /tmp/pinturas_catalogo

ENV PORT=8080
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

EXPOSE 8080

CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1
