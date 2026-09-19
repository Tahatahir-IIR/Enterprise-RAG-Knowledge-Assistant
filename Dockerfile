FROM python:3.11-slim

# OCR for scanned French/Arabic PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[embeddings,ocr,ui]"

COPY config ./config
COPY ui ./ui
COPY scripts ./scripts
COPY eval ./eval

EXPOSE 8000 8501
CMD ["uvicorn", "dossier.api:app", "--host", "0.0.0.0", "--port", "8000"]
