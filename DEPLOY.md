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

> ⚠️ **Set `ARTEMIS_ACCESS_TOKEN` before the URL leaves your hands.** Unset, the
> app is wide open and every build spends real money (search quota + Anthropic
> tokens). Generate one with:
> `python -c "import secrets; print(secrets.token_urlsafe(32))"`

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
fly secrets set ARTEMIS_ACCESS_TOKEN=your_generated_token \
                ANTHROPIC_API_KEY=your_anthropic_key \
                SERPER_API_KEY=your_serper_key \
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
   - `ARTEMIS_ACCESS_TOKEN` (secret — gates the whole app; set this)
   - `ANTHROPIC_API_KEY` (secret — enables the Claude extraction stages)
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
3. **Variables**: same `ARTEMIS_ACCESS_TOKEN`, `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `BRAVE_API_KEY`,
   `OPENCORPORATES_API_TOKEN`, and the four `ARTEMIS_*` paths above. Railway injects `$PORT`.
4. Keep **1 replica**.

---

## Local Docker (smoke test before deploying)

```bash
docker build -t artemis .
docker run -p 8080:8080 \
  -e ARTEMIS_ACCESS_TOKEN=your_token \
  -e ANTHROPIC_API_KEY=your_key \
  -e SERPER_API_KEY=your_key \
  -e BRAVE_API_KEY=your_key \
  -v "$(pwd)/data:/data" \
  artemis
# open http://localhost:8080
```

## Access control

Set `ARTEMIS_ACCESS_TOKEN` and the entire surface — UI and API — requires it.
Browsers get a sign-in page at `/login` that exchanges the token for an
HttpOnly session cookie (the cookie holds a derived value, never the token).
API clients send `Authorization: Bearer <token>`. Only `/healthz` and the login
routes stay open, so a load balancer can still probe a locked app.

Separately, and whether or not auth is on, the build endpoints are rate limited
per client (`ARTEMIS_BUILD_RATE_LIMIT`, default 12 per hour) and login attempts
are throttled (`ARTEMIS_LOGIN_RATE_LIMIT`, default 10 per 15 min). Reads are
never limited. Behind a managed host the client is identified from the
rightmost `X-Forwarded-For` hop; set `ARTEMIS_TRUST_PROXY_HEADERS=0` if you
ever expose the app directly, or the header can be forged to evade limits.

## Search provider configuration status

`ANTHROPIC_API_KEY`, `SERPER_API_KEY` and `BRAVE_API_KEY` are read once at startup from the
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
- **Extractor:** with `ANTHROPIC_API_KEY` set, the Claude **entity filter** and
  **relationship classifier** run on every build (both batched and cached 30
  days in the `/data` cache, so cost stays low). Page-level extraction stays on
  **spaCy** (the model is baked into the image) unless you opt in with
  `ARTEMIS_CLAUDE_EXTRACT=1` — that one is a Claude call per scraped page and is
  by far the most expensive setting in the app. With no key at all, every Claude
  stage no-ops and the build still works, just noisier. To save RAM you can set
  `ARTEMIS_SPACY_EXTRACT=0` (uses the heuristic instead).
- **Models:** the two batched stages (entity filter, relationship classifier)
  run on `ARTEMIS_CLAUDE_BATCH_MODEL` (default `claude-haiku-4-5` — narrow
  judgment calls on short strings, ~5x cheaper); page-level extraction runs on
  `ARTEMIS_CLAUDE_MODEL` (default `claude-opus-5`). Override either stage
  individually with `ARTEMIS_CLAUDE_FILTER_MODEL` /
  `ARTEMIS_CLAUDE_CLASSIFY_MODEL`. Raise the batch model to `claude-opus-5` if
  the filter starts dropping real people. `GET /status` reports which stages are
  live and whether a credential resolved — never the key itself.
- **Long requests:** every build endpoint (`/discover`, `/connect`,
  `/targets/search`) returns a `job_id` immediately and runs in the background,
  so nothing is held open against a host proxy's idle timeout. Poll
  `GET /jobs/{id}` for `pct`, `message`, `queue_position`, and the result;
  `POST /jobs/{id}/cancel` stops a job whether it is running or still queued.
  Jobs live in memory — a restart loses in-flight work, which is the accepted
  trade for having no queue infrastructure on a single-box deployment.
- **Concurrency:** `ARTEMIS_MAX_CONCURRENT_BUILDS` (default 2) builds run at
  once, `ARTEMIS_MAX_QUEUED_BUILDS` (default 8) may wait; past that a request
  gets an immediate 429 rather than joining a line it would time out in. Still
  **one worker** — the queue and job registry are in-process.
