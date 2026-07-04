# Artemis V2 — a single FastAPI app that also serves its own UI.
# It is long-running and stateful (SQLite on disk), so deploy it to a host with
# a PERSISTENT DISK (Fly.io / Render / Railway), never a serverless platform.
# Run exactly ONE worker: the app holds in-process state (the build lock and the
# per-session engine cache) that must be shared, not split across workers.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps first (better layer caching). spaCy model is the on-server extractor
# (Ollama isn't available in the container; the app auto-falls back to spaCy).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY app ./app
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# All SQLite state lives on the mounted volume at /data so it survives restarts.
# Override any of these at deploy time if your host mounts the disk elsewhere.
ENV ARTEMIS_DB_URL="sqlite:////data/artemis.db" \
    ARTEMIS_CACHE_DB="/data/artemis_cache.db" \
    ARTEMIS_GRAPH_DIR="/data/graphs" \
    ARTEMIS_CACHED_GRAPHS_DIR="/data/cached_graphs" \
    PORT=8080

EXPOSE 8080
ENTRYPOINT ["./docker-entrypoint.sh"]
