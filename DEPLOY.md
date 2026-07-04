# Deploying Artemis V2

Artemis is **one FastAPI app that also serves its UI**, and it is **long-running
and stateful** (SQLite on disk). Deploy it to a host with a **persistent disk** —
Fly.io, Render, or Railway. **Do not use a serverless platform** (Vercel/Lambda):
requests take minutes, and the filesystem must persist.

Two hard rules on any host:
1. **One worker only.** The app keeps in-process state (a build lock + a
   per-session engine cache). Never run multiple workers/replicas.
2. **All SQLite state on a mounted disk at `/data`** (cache, per-session graphs,
   Brave quota). The `ARTEMIS_*` env vars in the Dockerfile already point there.

Secrets are set as **platform env vars**, never committed (`.env` is gitignored).

> ⚠️ No auth/rate-limiting yet — keep the URL private. Anyone with the link can
> burn your Brave quota. Add a token before sharing widely.

---

## Fly.io (primary — `fly.toml` included)

```bash
# once: install + login
brew install flyctl        # or: curl -L https://fly.io/install.sh | sh
fly auth login

cd /path/to/ArtemisV2

# create the app (uses fly.toml; don't deploy yet)
fly launch --no-deploy --name artemisv2 --copy-config

# persistent disk for /data (3 GB is plenty; the cache is the biggest user)
fly volumes create artemis_data --size 3 --region sjc

# secrets (NOT in fly.toml / git)
fly secrets set BRAVE_API_KEY=your_brave_key \
                OPENCORPORATES_API_TOKEN=your_oc_token

fly deploy
fly open        # opens the UI
```

Notes:
- Change `primary_region` in `fly.toml` (and `--region`) to one near you.
- Logs: `fly logs`. SSH in: `fly ssh console`. Scale RAM: edit `[[vm]] memory`.

---

## Render (Docker web service + disk)

1. New → **Web Service** → connect the GitHub repo → **Docker** runtime
   (it uses the `Dockerfile`).
2. **Disks** → add a disk, mount path **`/data`**, size ~3 GB.
3. **Environment** → add:
   - `BRAVE_API_KEY`, `OPENCORPORATES_API_TOKEN` (secrets)
   - `ARTEMIS_DB_URL=sqlite:////data/artemis.db`
   - `ARTEMIS_CACHE_DB=/data/artemis_cache.db`
   - `ARTEMIS_GRAPH_DIR=/data/graphs`
   - `ARTEMIS_CACHED_GRAPHS_DIR=/data/cached_graphs`
4. Instances: **1** (do not scale out). Render sets `$PORT`; the entrypoint uses it.

A paid instance is required for a persistent disk.

---

## Railway (Docker + volume)

1. New Project → Deploy from GitHub repo (detects the `Dockerfile`).
2. Add a **Volume** mounted at **`/data`**.
3. **Variables**: same `BRAVE_API_KEY`, `OPENCORPORATES_API_TOKEN`, and the four
   `ARTEMIS_*` paths above. Railway injects `$PORT`.
4. Keep **1 replica**.

---

## Local Docker (smoke test before deploying)

```bash
docker build -t artemis .
docker run -p 8080:8080 \
  -e BRAVE_API_KEY=your_key \
  -v "$(pwd)/data:/data" \
  artemis
# open http://localhost:8080
```

## What runs where
- **Extractor:** Ollama isn't in the container, so it falls back to **spaCy**
  (the model is baked into the image). The deterministic junk filter still runs.
  To save RAM you can set `ARTEMIS_SPACY_EXTRACT=0` (uses the heuristic instead).
- **Long requests:** depth-1 Discover is quick; deep `connect` builds can run many
  minutes and may hit a host proxy's idle timeout. Prefer Discover for the beta;
  an async job model is the real fix (tracked as future work).
