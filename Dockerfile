# Artemis V2 — a single FastAPI app that also serves its own UI.
# It is long-running and stateful, so deploy it to a host that runs a persistent
# process (Fly.io / Render / Railway), never a serverless platform.
# Run exactly ONE worker: the app holds in-process state (the build lock and the
# per-session engine cache) that must be shared, not split across workers.
# The graph belongs in Postgres — set DATABASE_URL at deploy time (see DEPLOY.md).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps first (better layer caching). spaCy is the default page-level extractor;
# the Claude stages (entity filter, relationship classifier) switch on when
# ANTHROPIC_API_KEY is present in the environment at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY app ./app
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# File-backed state lives on the mounted volume at /data so it survives
# restarts. Override any of these at deploy time if your host mounts the disk
# elsewhere.
#
# ARTEMIS_DB_URL is deliberately NOT set here. It takes precedence over
# DATABASE_URL (see config._resolve_db_url), so baking a SQLite path into the
# image would silently override the Postgres database the host injects — the
# app would come up healthy, on the wrong database, with no error to notice.
# Unset, the graph follows DATABASE_URL, and falls back to a local SQLite file
# only when nothing at all is configured.
ENV ARTEMIS_CACHE_DB="/data/artemis_cache.db" \
    ARTEMIS_GRAPH_DIR="/data/graphs" \
    ARTEMIS_CACHED_GRAPHS_DIR="/data/cached_graphs" \
    PORT=8080

EXPOSE 8080
ENTRYPOINT ["./docker-entrypoint.sh"]
