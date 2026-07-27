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

# secrets (NOT in fly.toml / git) — both search providers are optional;
# whichever are set are picked up on next deploy, no code changes needed.
fly secrets set SERPER_API_KEY=your_serper_key \
                BRAVE_API_KEY=your_brave_key \
                OPENCORPORATES_API_TOKEN=your_oc_token

fly deploy
fly open        # opens the UI
```

Notes:
- Change `primary_region` in `fly.toml` (and `--region`) to one near you.
- Logs: `fly logs`. SSH in: `fly ssh console`. Scale RAM: edit `[[vm]] memory`.
- `fly secrets set` restarts the app with the new env vars injected directly —
  there is no build-time step and no GitHub Actions workflow involved. The
  running container reads `SERPER_API_KEY`/`BRAVE_API_KEY` straight from Fly's
  secret store on startup ([app/config.py](app/config.py)); end users never see
  or manage these keys; the UI just shows whether a provider is configured
  (see below).

---

## Render (Docker web service + disk)

1. New → **Web Service** → connect the GitHub repo → **Docker** runtime
   (it uses the `Dockerfile`).
2. **Disks** → add a disk, mount path **`/data`**, size ~3 GB.
3. **Environment** → add:
   - `SERPER_API_KEY`, `BRAVE_API_KEY`, `OPENCORPORATES_API_TOKEN` (secrets — all optional)
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
3. **Variables**: same `SERPER_API_KEY`, `BRAVE_API_KEY`, `OPENCORPORATES_API_TOKEN`, and the four
   `ARTEMIS_*` paths above. Railway injects `$PORT`.
4. Keep **1 replica**.

---

## Local Docker (smoke test before deploying)

```bash
docker build -t artemis .
docker run -p 8080:8080 \
  -e SERPER_API_KEY=your_key \
  -e BRAVE_API_KEY=your_key \
  -v "$(pwd)/data:/data" \
  artemis
# open http://localhost:8080
```

## Search provider configuration status

`SERPER_API_KEY` and `BRAVE_API_KEY` are read once at startup from the
platform's own secrets (`fly secrets set`, Render/Railway env vars, `docker run
-e`) — there is no in-app settings screen for entering them, and the running
app never returns their values to a client. `GET /status` reports each
provider's state (`ok` / `exhausted` / `invalid_key` / `not_configured`) so the
UI can tell whether *a* provider is configured, without ever seeing the key
itself. The home screen footer shows a "no search provider configured" notice
only when neither key is set (e.g. a bare local dev checkout); it disappears
the moment either secret is present, so end users never need to manage a key
that's already handled for them.

## What runs where
- **Extractor:** Ollama isn't in the container, so it falls back to **spaCy**
  (the model is baked into the image). The deterministic junk filter still runs.
  To save RAM you can set `ARTEMIS_SPACY_EXTRACT=0` (uses the heuristic instead).
- **Long requests:** depth-1 Discover is quick; deep `connect` builds can run many
  minutes and may hit a host proxy's idle timeout. Prefer Discover for the beta;
  an async job model is the real fix (tracked as future work).
