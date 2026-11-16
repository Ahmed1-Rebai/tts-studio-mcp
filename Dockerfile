ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG INSTALL_XTTS=false
ARG INSTALL_ASR=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TTS_APP_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    libespeak1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements-xtts.txt README.md main.py ./
COPY tts_app ./tts_app

# Install headless dependencies from pyproject.toml (excludes PyQt5 GUI libs)
RUN pip install --upgrade pip \
    && pip install . \
    && if [ "$INSTALL_XTTS" = "true" ]; then \
         pip install torch torchaudio torchcodec --index-url "$TORCH_INDEX_URL" && \
         pip install -r requirements-xtts.txt; \
       fi \
    && if [ "$INSTALL_ASR" = "true" ]; then \
         pip install faster-whisper; \
       fi \
    && mkdir -p /data /workspace /root/.local/share/tts

WORKDIR /workspace

CMD ["tts-mcp"]
