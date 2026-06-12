---
layout: default
title: Installation
---

# Installation

[← Back to Home](index.md)

## Requirements

- Python 3.11
- A Bluesky account with an **App Password** (Settings → App Passwords on Bluesky)
- An OpenAI API key (for AI seed generation in the admin panel)

## Local Development Setup

```powershell
# 1. Clone and enter directory
git clone https://github.com/osmankantarcioglu/BlueSky.git
cd BlueSky

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install the full NLP stack
pip install -r requirements-nlp.txt
pip install openai

# 4. Configure environment variables
copy .env.example .env
# Edit .env: set BSKY_HANDLE, BSKY_APP_PASSWORD, OPENAI_API_KEY
# Comment out DATABASE_URL to use local SQLite

# 5. Terminal 1 — Flask web server
python feed_generator/server.py

# 6. Terminal 2 — NLP worker (downloads BERTurk ~400MB on first run)
python data_collection/firehose_listener.py
```

Once both are running, open the admin panel at:

```
http://localhost:5000/admin
```

## Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `BSKY_HANDLE` | web + worker | Bluesky handle (e.g. `user.bsky.social`) |
| `BSKY_APP_PASSWORD` | web + worker | Bluesky **app password** |
| `FEED_DOMAIN` | web + worker | Public domain of the web service |
| `DATABASE_URL` | web + worker | PostgreSQL connection string (Railway auto-sets) |
| `DATABASE_PATH` | local dev | SQLite file path (default: `data/feeds.db`) |
| `OPENAI_API_KEY` | web + worker | GPT-4o seed generation in admin panel |
| `FLASK_SECRET_KEY` | web | Flask session secret (required for flash messages) |
| `HF_HOME` | worker | HuggingFace model cache path |
| `LLM_PROVIDER` | web | `openai` (default) or `anthropic` |

## Production Deployment (Railway)

The project runs as two Railway services sharing one PostgreSQL database:

- **`web`** — Flask + Waitress, serves AT Protocol endpoints and the admin panel
  - Start command: `waitress-serve --host=0.0.0.0 feed_generator.server:app`
- **`BlueSky` (worker)** — runs the firehose listener + NLP pipeline
  - Built from `Dockerfile.worker` (CPU PyTorch + `requirements-nlp.txt`)
  - Start command: `python data_collection/firehose_listener.py`
  - Mount a Railway Volume at `/hf_cache` to cache HuggingFace models (~440 MB BERTurk)

---

Next: [Usage Guide](usage.md)
