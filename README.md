# BlueSky Turkish Politics & Science Feed Generator

A custom feed generator for the [Bluesky](https://bsky.app) social network that curates Turkish-language posts about **politics** and **science** in real time. Built on the AT Protocol firehose, BERTurk NLP embeddings, PostgreSQL, and Flask — deployed on [Railway](https://railway.app).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Project Structure](#project-structure)
5. [Components](#components)
   - [Config](#1-configsettingspy)
   - [Database Models](#2-databasemodelspy)
   - [Firehose Listener](#3-data_collectionfirehose_listenerpy)
   - [Seed Discovery](#4-data_collectionseed_discoverypy)
   - [Turkish Embedder](#5-nlpembedderpy)
   - [Domain Classifier](#6-nlpdomain_classifierpy)
   - [Stance Detector](#7-nlpstance_detectorpy)
   - [NLP Pipeline](#8-nlppipelinepy)
   - [Feed Logic](#9-feed_generatorfeed_logicpy)
   - [Feed Server](#10-feed_generatorserverpy)
   - [Scripts](#11-scripts)
6. [NLP Classification Details](#nlp-classification-details)
7. [Ranking Algorithm](#ranking-algorithm)
8. [Deployment](#deployment)
   - [Local Development](#local-development)
   - [Railway Production](#railway-production)
9. [Environment Variables](#environment-variables)
10. [Database Schema](#database-schema)
11. [Key Parameters](#key-parameters)
12. [Performance Notes](#performance-notes)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This project creates two publicly accessible Bluesky custom feeds:

| Feed | Rkey | Description |
|------|------|-------------|
| Türkiye Siyaset | `turkiye-siyaset` | Turkish political posts — party news, parliament, elections |
| Türkiye Bilim | `turkiye-bilim` | Turkish science posts — research, academia, publications |

Both feeds are registered on Bluesky under the account `osmankantarcioglu.bsky.social` and served from a Railway-hosted Flask application. Posts are collected from a curated set of 270 seed users (politicians, academics, journalists) via the AT Protocol firehose and classified by a BERTurk-based NLP pipeline before being stored in a shared PostgreSQL database.

**Key design decisions:**
- Seed users give a high-quality, domain-relevant starting corpus instead of searching the entire Bluesky network
- NLP classification uses zero-shot centroid similarity — no labeled training data required
- Politics and science are mutually exclusive — every post gets exactly one label or is discarded
- Stance detection (alliance vs. opposition) is computed only for politics posts
- Two separate Railway services: a lightweight Flask server and a full ML worker

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   Bluesky Firehose                         │
│         (AT Protocol WebSocket — ~300 events/sec)          │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               FirehoseProcessor (Railway Worker)            │
│                                                             │
│  Stage 1: Event type filter  (only app.bsky.feed.post)     │
│  Stage 2: Fast pre-filter    (seed DID ∨ keyword match)    │
│  Stage 3: NLP pipeline       (language → embed → classify) │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     NLP Pipeline                            │
│                                                             │
│  ① Turkish language detection (langdetect ≥ 0.80)          │
│  ② BERTurk embedding (768-dimensional vector)              │
│  ③ Domain classification (cosine similarity to centroids)  │
│     └─ politics | science | other (discarded)              │
│  ④ Stance detection (politics only)                        │
│     └─ alliance | opposition | neutral                     │
│  ⑤ Feed score computation & DB write                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL Database (Railway)                  │
│                                                             │
│  tracked_users   posts   like_events                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│             Flask Feed Server (Railway Web)                 │
│                                                             │
│  /.well-known/did.json                                     │
│  /xrpc/app.bsky.feed.describeFeedGenerator                 │
│  /xrpc/app.bsky.feed.getFeedSkeleton                       │
│  /health                                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              Bluesky clients (web, mobile apps)
```

---

## Data Flow

1. **Firehose ingestion** — The AT Protocol firehose streams every create/delete event on Bluesky (~200-500 messages/second). The `FirehoseProcessor` subscribes via WebSocket, decodes CAR-encoded blocks, and extracts post records.

2. **Fast filtering** — Before any expensive NLP, two cheap checks run:
   - Is the author in the seed DID set? (O(1) hash lookup)
   - Does the post text contain at least one domain keyword? (O(k) linear scan over ~110 keywords)
   Any post passing either check goes into the NLP queue.

3. **NLP processing (batched)** — A background worker drains the queue in batches of up to 32 posts every 5 seconds:
   - Language detection rejects non-Turkish posts
   - BERTurk embedder converts text to a 768-dimensional semantic vector
   - Domain classifier computes cosine similarity to pre-built centroid vectors
   - Stance detector runs only on politics-classified posts
   - `feed_score` is computed and the post is upserted into the database

4. **Feed serving** — When a Bluesky client requests a feed, the Flask server queries PostgreSQL for posts matching the domain label, ordered by `feed_score`. It returns a list of AT URIs as a `FeedSkeleton`.

---

## Project Structure

```
BlueSky/
├── config/
│   ├── __init__.py
│   └── settings.py                  # All constants, thresholds, keyword lists
│
├── database/
│   ├── __init__.py
│   └── models.py                    # Peewee ORM — SQLite (dev) / PostgreSQL (prod)
│
├── data_collection/
│   ├── __init__.py
│   ├── firehose_listener.py         # Real-time AT Protocol stream consumer
│   └── seed_discovery.py            # Loads seed users from Excel → DB
│
├── nlp/
│   ├── __init__.py
│   ├── embedder.py                  # BERTurk sentence embeddings
│   ├── domain_classifier.py         # Politics vs Science (centroid similarity)
│   ├── stance_detector.py           # Alliance vs Opposition detection
│   └── pipeline.py                  # Orchestrates the full NLP workflow
│
├── feed_generator/
│   ├── __init__.py
│   ├── server.py                    # Flask HTTP server (AT Protocol endpoints)
│   └── feed_logic.py                # Feed ranking algorithm
│
├── scripts/
│   ├── build_domain_centroids.py    # Pre-builds centroid .npy files
│   └── publish_feed.py              # Registers feeds on Bluesky
│
├── data/                            # Runtime-generated, not in git
│   ├── feeds.db                     # SQLite (local dev only)
│   ├── centroids.npy                # Domain classifier vectors
│   └── stance_centroids.npy         # Stance detector vectors
│
├── Dockerfile.worker                # Railway worker image (ML + firehose)
├── Procfile                         # Railway web service start command
├── requirements.txt                 # Lightweight server deps (Railway web)
├── requirements-nlp.txt             # Full ML deps (Railway worker / local)
├── runtime.txt                      # Python 3.11
├── .env.example                     # Environment variable template
└── .gitignore
```

---

## Components

### 1. `config/settings.py`

Single source of truth for all system parameters. Everything is configurable via environment variables with sensible defaults.

**Credentials and identifiers:**
```python
BSKY_HANDLE       = os.getenv("BSKY_HANDLE")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")
FEED_DOMAIN       = os.getenv("FEED_DOMAIN", "yourdomain.com")
```

**Feed URIs (overridable via env for published DID:plc identifiers):**
```python
FEED_URI_POLITICS = os.getenv(
    "FEED_URI_POLITICS",
    f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-siyaset"
)
FEED_URI_SCIENCE = os.getenv(
    "FEED_URI_SCIENCE",
    f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-bilim"
)
```

**NLP model:**
```python
EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
EMBEDDING_DIM   = 768
```

**Classification thresholds:**
```python
DOMAIN_SIMILARITY_THRESHOLD = 0.30   # Min cosine similarity to assign a domain
STANCE_CONFIDENCE_THRESHOLD = 0.60   # Min confidence for stance classification
MIN_TURKISH_PROB            = 0.80   # langdetect threshold
```

**Keyword lists (excerpts):**

`POLITICS_KEYWORDS` — 60+ terms including party names (`akp`, `chp`, `mhp`, `hdp`), institutional terms (`tbmm`, `meclis`, `seçim`, `cumhurbaşkanı`), and political events.

`SCIENCE_KEYWORDS` — 50+ terms including academic venues (`nature`, `arxiv`, `dergi`), academic roles (`doçent`, `profösör`, `doktora`), and research activities (`makale`, `tez`, `araştırma`).

`ALLIANCE_KEYWORDS` / `OPPOSITION_KEYWORDS` — keyword sets for stance detection.

---

### 2. `database/models.py`

Peewee ORM models with automatic database switching based on `DATABASE_URL`:

```python
if DATABASE_URL:                      # Railway PostgreSQL
    db = PostgresqlDatabase(...)
else:                                 # Local SQLite
    db = SqliteDatabase(DATABASE_PATH)
```

**`TrackedUser`** — The 270 seed users loaded from Excel:

| Column | Type | Description |
|--------|------|-------------|
| `did` | PK CharField | AT Protocol decentralized identifier |
| `handle` | CharField | e.g. `user.bsky.social` |
| `display_name` | CharField | Full name |
| `party` | CharField | e.g. `AKP`, `CHP`, `MHP` |
| `stance` | CharField | `alliance` / `opposition` / `unknown` |
| `domain` | CharField | `politics` / `science` / `both` |
| `source` | CharField | `csv` / `starter_pack` / `search` |
| `is_active` | BooleanField | Whether to track this user |

**`Post`** — Every classified post that passed NLP filtering:

| Column | Type | Description |
|--------|------|-------------|
| `uri` | PK CharField | AT URI — e.g. `at://did:xxx/app.bsky.feed.post/xxx` |
| `cid` | CharField | Content hash (for deduplication) |
| `author_did` | CharField | Post author DID |
| `author_handle` | CharField | Post author handle |
| `text` | TextField | Full post text |
| `domain_label` | CharField | `politics` or `science` |
| `stance_label` | CharField | `alliance` / `opposition` / `neutral` |
| `domain_score` | FloatField | Cosine similarity to domain centroid |
| `stance_score` | FloatField | Stance confidence (0–1) |
| `embedding` | TextField | JSON-encoded 768-dim BERTurk vector |
| `created_at` | DateTimeField | When the post was created on Bluesky |
| `indexed_at` | DateTimeField | When we processed it |
| `language` | CharField | Detected language (usually `tr`) |
| `like_count` | IntegerField | Engagement metric |
| `repost_count` | IntegerField | Engagement metric |
| `reply_count` | IntegerField | Engagement metric |
| `feed_score` | FloatField | Combined ranking score (indexed) |

**`LikeEvent`** — Like events captured from the firehose for engagement updates.

---

### 3. `data_collection/firehose_listener.py`

The core ingestion service. Connects to `wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos` and processes every commit event in real time.

**Class `FirehoseProcessor`:**

```
FirehoseProcessor
├── setup()
│   ├── Load seed DIDs from TrackedUser table
│   └── Initialize NLP pipeline (load or build centroids)
│
├── on_message_handler(message)          ← called per firehose event
│   ├── Decode CAR block (Merkle DAG)
│   ├── Extract post record + metadata
│   ├── Check: event type == app.bsky.feed.post?
│   ├── Check: is_seed OR has_keyword?
│   └── Push to post_queue
│
├── _process_queue_worker()              ← background thread
│   ├── Drain queue every 5s or 32 posts
│   └── Send batch to NLPPipeline
│
├── _stats_reporter()                    ← background thread
│   └── Print throughput every 60s
│
└── start()
    ├── Spin up worker threads
    └── Subscribe to firehose (reconnects on failure)
```

**Statistics logged every 60 seconds:**
```
[Stats] received=67398 queued=2358 saved=142
Batch done: 32 processed | total queued=2389 saved=143
```

**Thread safety:** The `post_queue` is a `collections.deque(maxlen=1000)` — fast O(1) appends and pops, bounded to prevent memory overflow during backpressure.

---

### 4. `data_collection/seed_discovery.py`

One-time setup script that populates `tracked_users` from an Excel spreadsheet.

**Input:** `bsky_manual_minimal.xlsx` — 273 rows, each with a Bluesky handle and metadata (party, domain, display name).

**Process:**
1. Read Excel with pandas, filter rows that have a `bsky_handle` value
2. Authenticate with Bluesky using app password
3. For each row: resolve handle → DID via `client.resolve_handle()`
4. Infer stance from party:
   - Alliance: `AKP`, `MHP`
   - Opposition: `CHP`, `HDP`, `DEM`, `İYİ Parti`, `DEVA`, `ZAFER`, `TİP`
5. Upsert into `TrackedUser` (safe to re-run)
6. Rate-limit: 0.5 second delay between API calls

**Result:** 270 users saved (3 handles failed to resolve due to account changes).

---

### 5. `nlp/embedder.py`

**Class `TurkishEmbedder`:**

Wraps `sentence-transformers` with the BERTurk NLI/STS model fine-tuned for semantic similarity in Turkish.

**Model:** `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`
- Architecture: BERT Base (12 layers, 12 heads, 768 hidden)
- Vocabulary: 32,000 Turkish cased tokens
- Trained on: NLI + STS-B tasks in Turkish
- Output: Mean-pooled token embeddings, normalized to unit length
- Download size: ~440 MB (cached after first run)

**Text preprocessing pipeline:**
```
raw text
  → remove URLs (http/https)
  → remove @mentions
  → strip # from hashtags (keep word)
  → collapse multiple whitespace
  → truncate to 512 characters (BERT token limit)
```

**Key methods:**
- `embed(text)` → `(768,)` numpy array
- `embed_batch(texts, batch_size=32)` → `(N, 768)` numpy matrix
- `vector_to_json(vector)` → JSON string for DB storage
- `json_to_vector(json_str)` → numpy array

**Device selection:** Automatically uses GPU (`cuda`) if available, falls back to CPU. On Railway CPU-only deployment, embedding takes ~50ms per post.

---

### 6. `nlp/domain_classifier.py`

**Class `DomainClassifier`:**

Zero-shot classification using centroid similarity — no labeled training data required.

**Algorithm:**

```
For each incoming post:

  1. Compute BERTurk embedding: e ∈ ℝ⁷⁶⁸

  2. For each domain d ∈ {politics, science}:
       sim(d) = cosine_similarity(e, centroid_d)

  3. Filter: above = {d : sim(d) ≥ DOMAIN_SIMILARITY_THRESHOLD}

  4. If |above| ≥ 1:
       return argmax(sim), max(sim)   ← always ONE domain, never "both"

  5. Else:
       return 'other', 0.0            ← discard this post
```

**Centroid construction:**

Each centroid is the mean embedding of ~45 representative seed sentences for that domain. Sentences are written to resemble actual Turkish tweets (colloquial, short, hashtag-heavy):

*Politics examples:*
- `"Erdoğan bugün TBMM'de konuştu, muhalefet sert tepki verdi"`
- `"CHP'nin yeni adayı açıklandı, sosyal medya yıkıldı"`
- `"Belediye başkanı görevden alındı, kayyum atandı"`
- `"Meclis gündeminde anayasa tartışması var"`

*Science examples:*
- `"Yeni makalemiz arXiv'e yüklendi, link biyoda"`
- `"Nature'da yayınlanan çalışmamız çok ilgi gördü"`
- `"Doktora savunmam geçti! Çok mutluyum"`
- `"TÜBİTAK projesine kabul edildim, heyecanlıyım"`

**Sample similarity scores (empirical):**
| Post | pol sim | sci sim | Label |
|------|---------|---------|-------|
| `"Erdogan bugün TBMM'de konuştu"` | 0.348 | 0.242 | politics |
| `"Yeni anayasa mecliste oylanacak"` | 0.495 | 0.325 | politics |
| `"Yapay zeka araştırmamız Nature'da"` | 0.323 | 0.601 | science |
| `"İklim değişikliği yeni bulgular"` | 0.440 | 0.534 | science |
| `"Bugün hava güzel"` | 0.201 | 0.201 | other |

**Centroid persistence:** Saved to `data/centroids.npy` as a dict `{domain: centroid_vector}`. If the file is missing at startup (e.g., fresh Railway deploy), centroids are rebuilt from keywords automatically and then saved.

---

### 7. `nlp/stance_detector.py`

Detects political stance only for posts already classified as `politics`.

**Class `StanceDetectorV1`** (production, keyword + centroid):

```
For each politics post:

  1. Compute cosine similarity to alliance_centroid and opposition_centroid

  2. If max(sim) ≥ STANCE_CONFIDENCE_THRESHOLD (0.60):
       return argmax(sim), max(sim)

  3. Else: keyword fallback
       count_alliance  = matches in ALLIANCE_KEYWORDS
       count_opposition = matches in OPPOSITION_KEYWORDS
       if count_alliance > count_opposition: return 'alliance'
       if count_opposition > count_alliance: return 'opposition'
       else: return 'neutral'
```

*Alliance centroid built from sentences like:*
- `"Cumhur ittifakı ülkenin istikrarı için önemli adımlar atıyor"`
- `"Savunma sanayisinde büyük başarılar elde ettik"`

*Opposition centroid built from sentences like:*
- `"Muhalefet halkın ekonomik sıkıntılarına dikkat çekiyor"`
- `"CHP yeni reform önerilerini paylaştı"`

**Class `StanceDetectorV2`** (advanced, fine-tuned):
- Requires 500+ manually labeled posts per class
- Fine-tunes BERTurk as a 3-class classifier
- Needs GPU for practical training time (~30 minutes)
- Documented in `scripts/fine_tune_stance.py`

---

### 8. `nlp/pipeline.py`

**Class `NLPPipeline`:**

Orchestrates the full NLP workflow. Called by `FirehoseProcessor` with batches of raw post data.

```python
def load_models(centroid_path, stance_path):
    # 1. Initialize TurkishEmbedder (loads BERTurk)
    # 2. Initialize DomainClassifier
    #    - Try to load centroids from .npy file
    #    - If missing: build from keywords, then save
    # 3. Initialize StanceDetector
    #    - Same fallback logic for stance centroids

def is_turkish(text) -> bool:
    # langdetect.detect(text) returns language code
    # Returns True if detected language is 'tr' with confidence ≥ MIN_TURKISH_PROB

def process_post(uri, cid, author_did, author_handle, text, created_at):
    # 1. Turkish check → return None if not Turkish
    # 2. Embed text → 768-dim vector
    # 3. Classify domain → 'politics' | 'science' | 'other'
    #    Return None if 'other' (post not saved)
    # 4. If 'politics': detect stance
    # 5. Compute initial feed_score
    # 6. Upsert Post record to database
    # 7. Return Post instance
```

---

### 9. `feed_generator/feed_logic.py`

The ranking algorithm that determines the order posts appear in the feed.

**Feed score formula:**

```
feed_score = (nlp_weight * domain_score + engagement_weight * engagement_norm) * recency_boost

Where:
  nlp_weight        = 0.5
  engagement_weight = 0.3
  domain_score      = cosine similarity to domain centroid (0–1)
  engagement_norm   = log1p(likes + 2*reposts + 0.5*replies) / 10
  recency_boost     = linear interpolation [0.5, 1.0] over 48 hours
```

**Recency boost:**
- Post age = 0 hours → boost = 1.0 (full weight)
- Post age = 24 hours → boost = 0.75
- Post age = 48+ hours → boost = 0.5 (floor)

This means a fresh post with moderate NLP score will outrank an old post with a perfect NLP score — keeping feeds timely.

**Pagination:** Uses cursor-based pagination. The cursor is the ISO 8601 timestamp of the last returned post's `indexed_at`. Each request returns posts created before that cursor, enabling infinite scroll.

**`get_feed_posts(domain, cursor, limit)`:**
```python
query = (
    Post.select()
    .where(Post.domain_label == domain)   # strict domain — no 'both'
    .order_by(Post.feed_score.desc(), Post.created_at.desc())
    .limit(MAX_FEED_POSTS)
)
```

---

### 10. `feed_generator/server.py`

Flask application implementing the AT Protocol feed generator specification.

**Endpoints:**

`GET /.well-known/did.json`
```json
{
  "@context": ["https://www.w3.org/ns/did/v1"],
  "id": "did:web:web-production-77bc8f.up.railway.app",
  "service": [{
    "id": "#bsky_fg",
    "type": "BskyFeedGenerator",
    "serviceEndpoint": "https://web-production-77bc8f.up.railway.app"
  }]
}
```

`GET /xrpc/app.bsky.feed.describeFeedGenerator`
```json
{
  "did": "did:web:...",
  "feeds": [
    {"uri": "at://did:plc:.../app.bsky.feed.generator/turkiye-siyaset"},
    {"uri": "at://did:plc:.../app.bsky.feed.generator/turkiye-bilim"}
  ]
}
```

`GET /xrpc/app.bsky.feed.getFeedSkeleton?feed=<uri>&cursor=<ts>&limit=<n>`
```json
{
  "cursor": "2024-01-15T10:30:00.000Z",
  "feed": [
    {"post": "at://did:xxx/app.bsky.feed.post/aaa"},
    {"post": "at://did:yyy/app.bsky.feed.post/bbb"}
  ]
}
```

`GET /health`
```json
{"status": "ok", "posts_in_db": 1234}
```

**Database connection lifecycle:** One connection per request (opened in `before_request`, closed in `teardown_request`) using Peewee's `database.connect(reuse_if_open=True)`.

**Startup:** Creates the `data/` directory and all DB tables if they don't exist — safe to run on a fresh Railway deployment.

**Production server:** `waitress-serve` (not Flask dev server) as specified in `Procfile`.

---

### 11. Scripts

**`scripts/publish_feed.py`**

One-time script run after the feed server is publicly accessible. Authenticates with Bluesky and creates two `app.bsky.feed.generator` records:

```
turkiye-siyaset → "Türkiye Siyaset"   Description: Turkish political discourse feed
turkiye-bilim   → "Türkiye Bilim"     Description: Turkish science and research feed
```

The resulting AT URIs (e.g. `at://did:plc:xxx/app.bsky.feed.generator/turkiye-siyaset`) are stored in `.env` as `FEED_URI_POLITICS` and `FEED_URI_SCIENCE`.

**`scripts/build_domain_centroids.py`**

Standalone script to pre-compute centroid vectors and save them to `data/centroids.npy`. Includes sanity checks to verify that known politics/science/noise sentences are classified correctly. The firehose listener also builds centroids on startup if the file is missing, so this script is optional but useful for inspecting centroid quality before deployment.

---

## NLP Classification Details

### Why Zero-Shot Centroid Similarity?

Training a supervised classifier would require hundreds of manually labeled Turkish posts per category. Instead, we use the fact that BERTurk embeddings encode semantic meaning — similar texts cluster together in embedding space. By computing the "center" (centroid) of a representative set of politics and science sentences, we get a reference point and classify new posts by proximity.

**Advantages:**
- No labeled data required
- Easy to update by adding/removing seed sentences
- Works well even with Turkish social media language (abbreviations, hashtags, slang)

**Threshold (0.30):** Empirically determined from testing with real Bluesky posts. Genuine politics/science posts score 0.35–0.60+; irrelevant posts score below 0.25.

### Why No Keyword Fallback in the Classifier?

An earlier version used keyword matching as a fallback when no domain scored above the threshold. This caused false positives — a post mentioning "üniversite" (university) in a personal context (e.g. "üniversite arkadaşıyla buluştum") would be classified as science. The keyword fallback was removed entirely; now only centroid similarity determines domain.

### Strict Domain Assignment

A post can only belong to **one** domain. If both `politics` and `science` scores are above the threshold, the higher-scoring domain wins. The label `'both'` is never used. This ensures each post appears in exactly one feed.

---

## Ranking Algorithm

Posts in each feed are sorted by `feed_score`, a weighted combination of:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| NLP domain similarity | 50% | Ensures topical relevance |
| Engagement (likes, reposts, replies) | 30% | Surfaces quality content |
| Recency boost | Variable | Keeps feed fresh |

The recency boost is a multiplier (0.5–1.0) that decays linearly over 48 hours, preventing old posts from permanently dominating.

Feed scores are periodically refreshed as engagement counts change (called every 5 minutes via `refresh_feed_scores()`).

---

## Deployment

### Local Development

**Prerequisites:** Python 3.11, pip, PostgreSQL or SQLite

```bash
# 1. Clone repo and enter directory
git clone https://github.com/osmankantarcioglu/BlueSky.git
cd BlueSky

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install NLP dependencies (includes ML libraries)
pip install -r requirements-nlp.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your Bluesky credentials and DATABASE_URL

# 5. Initialize database
python -c "from database.models import create_tables; create_tables()"

# 6. Load seed users (requires bsky_manual_minimal.xlsx)
python data_collection/seed_discovery.py

# 7. (Optional) Pre-build centroids
python scripts/build_domain_centroids.py

# 8. Start the firehose listener
python data_collection/firehose_listener.py

# 9. In a separate terminal, start the feed server
waitress-serve --host=0.0.0.0 --port=8080 feed_generator.server:app
```

### Railway Production

The production setup uses two separate Railway services in the same project, sharing one PostgreSQL database.

**Service 1: Feed Server (web)**
- Builder: Railpack (default)
- `requirements.txt`: lightweight — no torch or transformers
- Start command: defined in `Procfile`
- Needs: `DATABASE_URL`, `BSKY_HANDLE`, `BSKY_APP_PASSWORD`, `FEED_DOMAIN`, `FEED_URI_POLITICS`, `FEED_URI_SCIENCE`
- Healthcheck path: `/health`

**Service 2: NLP Worker (firehose listener)**
- Builder: Dockerfile → `Dockerfile.worker`
- `Dockerfile.worker` installs CPU-only PyTorch + sentence-transformers
- Start command: `python data_collection/firehose_listener.py` (in CMD)
- Restart policy: **Always** (reconnects to firehose if disconnected)
- Needs: same env vars as web service
- Volume: mount at `/hf_cache` for HuggingFace model cache (prevents re-download on restart)

**`Dockerfile.worker` highlights:**
```dockerfile
FROM python:3.11-slim
# CPU-only PyTorch → much smaller than GPU build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-nlp.txt .
RUN pip install -r requirements-nlp.txt
ENV HF_HOME=/hf_cache
CMD ["python", "data_collection/firehose_listener.py"]
```

**One-time setup (run locally after server is deployed):**
```bash
# Publish feeds to Bluesky (server must be live at FEED_DOMAIN)
python scripts/publish_feed.py
# Copy the printed feed URIs into Railway env vars
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BSKY_HANDLE` | Yes | Your Bluesky handle (e.g. `user.bsky.social`) |
| `BSKY_APP_PASSWORD` | Yes | Bluesky app password (not your login password) |
| `FEED_DOMAIN` | Yes | Public domain of your Railway web service |
| `DATABASE_URL` | Prod | PostgreSQL connection string (Railway provides this) |
| `DATABASE_PATH` | Dev | SQLite file path (default: `data/feeds.db`) |
| `FEED_URI_POLITICS` | Yes | AT URI for the politics feed (from `publish_feed.py`) |
| `FEED_URI_SCIENCE` | Yes | AT URI for the science feed (from `publish_feed.py`) |
| `HF_HOME` | Worker | HuggingFace cache directory (use Railway volume path) |
| `TRANSFORMERS_CACHE` | Worker | Same as `HF_HOME` (legacy env var) |

---

## Database Schema

### `tracked_users`
```sql
CREATE TABLE tracked_users (
    did          VARCHAR PRIMARY KEY,
    handle       VARCHAR NOT NULL,
    display_name VARCHAR,
    party        VARCHAR,
    stance       VARCHAR DEFAULT 'unknown',   -- alliance|opposition|unknown
    domain       VARCHAR DEFAULT 'politics',  -- politics|science|both
    source       VARCHAR DEFAULT 'csv',
    created_at   TIMESTAMP DEFAULT NOW(),
    is_active    BOOLEAN DEFAULT TRUE
);
```

### `posts`
```sql
CREATE TABLE posts (
    uri          VARCHAR PRIMARY KEY,
    cid          VARCHAR NOT NULL,
    author_did   VARCHAR NOT NULL,
    author_handle VARCHAR,
    text         TEXT NOT NULL,
    domain_label VARCHAR NOT NULL,            -- politics|science
    stance_label VARCHAR DEFAULT 'neutral',   -- alliance|opposition|neutral
    domain_score FLOAT DEFAULT 0.0,
    stance_score FLOAT DEFAULT 0.0,
    embedding    TEXT,                         -- JSON: [768 floats]
    created_at   TIMESTAMP NOT NULL,
    indexed_at   TIMESTAMP DEFAULT NOW(),
    language     VARCHAR DEFAULT 'tr',
    like_count   INTEGER DEFAULT 0,
    repost_count INTEGER DEFAULT 0,
    reply_count  INTEGER DEFAULT 0,
    feed_score   FLOAT DEFAULT 0.0
);

CREATE INDEX idx_posts_domain_created ON posts (domain_label, created_at);
CREATE INDEX idx_posts_feed_score     ON posts (feed_score);
CREATE INDEX idx_posts_author         ON posts (author_did);
```

### `like_events`
```sql
CREATE TABLE like_events (
    uri        VARCHAR NOT NULL,
    liker_did  VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Key Parameters

| Parameter | Value | Where set |
|-----------|-------|-----------|
| `DOMAIN_SIMILARITY_THRESHOLD` | 0.30 | `config/settings.py` |
| `STANCE_CONFIDENCE_THRESHOLD` | 0.60 | `config/settings.py` |
| `MAX_FEED_POSTS` | 200 | `config/settings.py` |
| `FEED_CACHE_TTL` | 300 s | `config/settings.py` |
| `MIN_TURKISH_PROB` | 0.80 | `config/settings.py` |
| Firehose batch size | 32 posts | `firehose_listener.py` |
| Queue drain interval | 5 s | `firehose_listener.py` |
| Stats report interval | 60 s | `firehose_listener.py` |
| Seed user API delay | 0.5 s | `seed_discovery.py` |
| Recency window | 48 h | `feed_logic.py` |
| Ranking: NLP weight | 0.5 | `feed_logic.py` |
| Ranking: engagement weight | 0.3 | `feed_logic.py` |

---

## Performance Notes

| Metric | Value |
|--------|-------|
| Firehose throughput | ~200–500 events/sec |
| Post queue limit | 1,000 (deque maxlen) |
| NLP embedding time (CPU) | ~50 ms/post |
| NLP embedding time (GPU) | ~5 ms/post |
| Batch size (NLP) | 32 posts |
| BERTurk model size | ~440 MB |
| Centroid build time | ~5–10 s |
| Feed query time | <100 ms |
| PostgreSQL connection | Peewee autorollback mode |

---

## Troubleshooting

**`saved=0` in firehose stats**
The threshold may be too high for the current centroid quality. Check actual similarity scores:
```python
from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier
emb = TurkishEmbedder()
clf = DomainClassifier(emb)
clf.load_centroids('data/centroids.npy')
print(clf.classify("Meclis bugün toplandı"))
```
If politics posts score < 0.30, lower `DOMAIN_SIMILARITY_THRESHOLD` or rebuild centroids with more tweet-like seed sentences.

**`FileNotFoundError: data/centroids.npy`**
The `data/` directory does not exist. This is fixed in `save_centroids()` — it now calls `os.makedirs()` before writing. Pull the latest version and redeploy.

**`DATABASE_URL` connection refused**
Make sure you are using the **public** Railway PostgreSQL URL (`gondola.proxy.rlwy.net:12946`), not the internal hostname (`postgres.railway.internal`) which is only accessible within the Railway private network.

**Feed shows "This feed is empty" on Bluesky**
Either no posts have been saved yet (give the firehose listener time to collect data), or the `FEED_URI_*` env vars don't match the URIs registered via `publish_feed.py`. Check `/xrpc/app.bsky.feed.getFeedSkeleton?feed=<uri>` directly on the Railway domain.

**Railway build timeout**
The web service uses `requirements.txt` (no torch). The worker service uses `Dockerfile.worker` which installs CPU-only torch — significantly smaller than the GPU build. Never add torch to `requirements.txt`.

**Model re-downloaded on every worker restart**
Mount a Railway Volume at `/hf_cache` and set `HF_HOME=/hf_cache` in worker env vars. After the first run, the ~440 MB BERTurk model is cached on the volume.

---

## Feed URIs

| Feed | AT URI |
|------|--------|
| Türkiye Siyaset | `at://did:plc:tl4dbarzqear47dehrsgtlvr/app.bsky.feed.generator/turkiye-siyaset` |
| Türkiye Bilim | `at://did:plc:tl4dbarzqear47dehrsgtlvr/app.bsky.feed.generator/turkiye-bilim` |

Feed server: `https://web-production-77bc8f.up.railway.app`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Protocol | AT Protocol (atproto 0.0.55) |
| NLP embeddings | BERTurk (`emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`) |
| ML framework | PyTorch (CPU) + sentence-transformers + scikit-learn |
| Language detection | langdetect |
| ORM | Peewee |
| Database (dev) | SQLite |
| Database (prod) | PostgreSQL (Railway) |
| Web framework | Flask + Waitress |
| Deployment | Railway (web + worker services) |
| Container | Docker (worker only) |
| Language | Python 3.11 |
