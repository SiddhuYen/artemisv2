# Deploying Artemis V2

Artemis is **one FastAPI app that also serves its UI**, and it is **long-running
and stateful**. Deploy it to a host that can run a persistent process —
Fly.io, Render, or Railway. **Do not use a serverless platform** (Vercel/Lambda):
requests take minutes.

Three hard rules on any host:
1. **One worker only.** The app keeps in-process state (a build lock + a
   per-session engine cache). Never run multiple workers/replicas.
2. **The graph goes in Postgres**, via `DATABASE_URL`. Not SQLite — see
   [Database](#database) for why this is not a preference.
3. **The search cache on a mounted disk at `/data`.** It is a local file cache
   of fetched pages; losing it costs real money in re-bought search results,
   but nothing is corrupted if it goes.

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

Easiest path: **New → Blueprint** and point it at this repo — `render.yaml`
declares the service, the disk and every variable below, prompting only for the
secrets. To do it by hand instead:

1. New → **Web Service** → connect the GitHub repo → **Docker** runtime
   (it uses the `Dockerfile`).
2. **Disks** → add a disk, mount path **`/data`**, size ~3 GB.
3. **Environment** → add:
   - `ARTEMIS_ACCESS_TOKEN` (secret — gates the whole app; set this)
   - `DATABASE_URL` (secret — your Supabase URI; see [Database](#database))
   - `ANTHROPIC_API_KEY` (secret — enables the Claude extraction stages)
   - `SERPER_API_KEY`, `BRAVE_API_KEY`, `OPENCORPORATES_API_TOKEN` (secrets — all optional)
   - `ARTEMIS_CACHE_DB=/data/artemis_cache.db`
   - `ARTEMIS_GRAPH_DIR=/data/graphs`
   - `ARTEMIS_CACHED_GRAPHS_DIR=/data/cached_graphs`
4. Instances: **1** (do not scale out). Render sets `$PORT`; the entrypoint uses it.

A paid instance is required for a persistent disk. If you skip the disk, the
app still runs correctly — the graph is in Postgres — but the search cache
resets on every deploy and you re-pay for results you already bought.

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

## Database

**A deployment must use Postgres.** SQLite remains the zero-setup default for
local dev and the test suite, and the schema is identical on both — but it is
not a safe choice for a deployed instance, for two independent reasons.

**1. Concurrency, which is a correctness problem here, not a speed one.**
Expansion researches several people at once (`EXPAND_NODE_CONCURRENCY`,
doubled again by `/connect` walking both endpoints simultaneously), each on its
own session. SQLite allows exactly one writer at a time and serializes the rest
behind a single file lock. When a writer waited past the busy timeout, the node
being written was **dropped from the graph** — and a dropped node is silent
data loss: `/connect` then reports "no path" between two people who really are
connected. This was diagnosed live, not theorised: four of five frontier nodes
were lost in a single run. Postgres has row-level locking and MVCC, so
unrelated concurrent writes don't contend at all.

**2. Durability.** A disk on Render or Fly is tied to one paid instance.
Postgres survives instance replacement, plan changes and redeploys, can be
backed up, and can be inspected with `psql` while the app is running.

### Wiring it up

Set **`DATABASE_URL`** — the variable every managed host and Postgres add-on
populates automatically. Attach a database and the app finds it; there is no
Artemis-specific database configuration to do. `ARTEMIS_DB_URL` still takes
precedence when set, so a local override doesn't require unsetting the
platform's own variable.

With **Supabase** (Project → Connect → URI):

```
DATABASE_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres
```

Three things worth knowing:

- **Use the direct connection on port 5432**, not the transaction pooler on
  6543. Graph writes rely on `SAVEPOINT`s inside multi-statement transactions,
  which is session-level behaviour that a transaction pooler does not promise.
- **A `postgres://` string is fine.** Supabase, Render and Heroku all still
  print that scheme; SQLAlchemy 1.4+ rejects it, so the app rewrites it
  (`config._normalize_db_url`). Paste the string as given.
- **The free tier pauses when idle.** A paused project makes the app hang on
  connect rather than fail fast — worth knowing before you debug it as an app
  bug.

The schema builds itself via `Base.metadata.create_all` on first connect. There
is **no migration step**, and additive column migrations
([app/db.py](app/db.py)) run against either backend.

### What stays on SQLite

`ARTEMIS_CACHE_DB` — the provider-response cache. It is a raw-`sqlite3`
key/value store of fetched pages, entirely separate from the graph, and local
file access beats a network round-trip on every hit. Leave it pointed at the
mounted disk.

`ARTEMIS_BOARDS_DB_URL` is best left unset: on Postgres, boards default to the
**same** database as the graph, so they inherit its durability. The separate
file only ever existed to keep board autosaves from contending with the graph
for SQLite's single write lock — a problem Postgres does not have.

### Working as a team on one shared database

Collaborators can point their local checkouts at the same Supabase project and
build one shared graph. `cp .env.example .env`, then fill in `DATABASE_URL` and
the API keys. Pass the connection string through a password manager — it is a
credential, and it is not in git.

Three things to know before you do this:

**The CLI no longer wipes a shared graph.** `python -m app.cli "Some Name"`
resets the graph by default (`--keep` is opt-in) — which is fine against a
private local file, and catastrophic against a team database, where it would
delete everyone's work. On a shared database the CLI now **accumulates by
default** and says so; wiping requires an explicit `--force-reset`.
`reset_public_graph` refuses outright unless forced, so the same protection
covers every other caller — including `add-org-network`, which used to clear
the public graph as routine scratch cleanup.

**There is no per-person isolation.** One accumulating graph is the design —
it is what lets `/connect` route through people an earlier run discovered —
but it also means everyone sees everyone's noise, and a name someone typo'd
is in your results too. For solo experiments, leave `DATABASE_URL` blank and
work against your own local SQLite file.

**Everyone's keys spend real money.** Each build costs search credits and
Anthropic tokens. Sharing one set of keys means one bill with no attribution;
separate keys per person cost the same in total but tell you who spent what.

Alternatively, for collaborators who only need to *use* Artemis rather than
develop it: give them the deployed URL and `ARTEMIS_ACCESS_TOKEN` instead. No
setup, no credentials on their machines, and no way to run the destructive
commands at all.

### Local Postgres (optional)

To run the deployment backend locally:

```bash
brew install postgresql@16
brew services start postgresql@16
createdb artemis
```

Then in `.env` (gitignored): `ARTEMIS_DB_URL=postgresql://localhost/artemis`.
To go back to SQLite, delete that line. `psycopg2-binary` is already in
`requirements.txt`.

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
