---
layout: default
title: Usage
---

# Usage

[← Back to Home](index.md)

## Admin Panel

The admin panel is the main interface for creating and managing feeds. Once the server is running, open:

```
http://localhost:5000/admin       # local
https://web-production-77bc8f.up.railway.app/admin   # production
```

### Dashboard (`/admin/`)

- Stats bar: total feeds, active feeds, total posts
- Feed cards show status (Active / Stopped), keyword count, post count, and centroid status
- Per-card actions: ⏸ Stop · ▶ Start · ⚙ Details · 🗑 Delete

### Creating a New Feed (`/admin/feeds/new`)

**Step 1 — AI Generation**

1. Enter a topic (e.g. "FIFA World Cup 2026") and select a language
2. Click **Generate Seeds with AI** — GPT-4o produces:
   - 100+ keywords (used for the firehose pre-filter)
   - 50+ social-media-style seed sentences (used to build the topic centroid)
3. Review the generated keyword cloud and sentence preview

**Step 2 — Configuration**

1. Set a **Feed ID** — a URL-safe slug, max **15 characters** (Bluesky `rkey` limit)
2. Set the display name, description, language, embedding model, and similarity threshold
3. Toggle **"Publish to Bluesky automatically"** if you want the feed registered immediately
4. Submit — the feed is saved to the database and the worker picks it up within 60 seconds

### Feed Detail (`/admin/feeds/<id>`)

- Feed name, status badges, and AT URI
- Stats: Feed ID, similarity threshold, keyword count, seed sentence count
- Animated keyword cloud and seed sentence list
- Recent posts table with score bars
- Actions: Stop/Start, Rebuild Centroid, Publish to Bluesky (if not yet published), Delete

## How Posts Get Matched

```
Firehose post arrives
   │
   ├─ 1. Keyword pre-filter (fast, O(K))
   │      Does the text contain any feed keyword?
   │
   ├─ 2. Language check
   │      Does the post language match the feed's language setting?
   │
   └─ 3. Cosine similarity
          similarity(embed(post), feed.centroid) ≥ feed.similarity_threshold
          → if true, the post is saved and linked to the feed
```

## Feed Ranking

Posts shown in a feed are ranked by:

| Factor | Weight |
|--------|--------|
| NLP similarity score | 50% |
| Engagement (likes, reposts, replies) | 30% |
| Recency | Decays linearly over 48h |

## Troubleshooting

**"Centroid Pending" on feed detail**
The web server doesn't run NLP — the worker builds the centroid on its next 60-second poll cycle.

**"Publish to Bluesky" fails**
Verify `BSKY_HANDLE`, `BSKY_APP_PASSWORD`, and `FEED_DOMAIN` are set on the **web** service, and that the app password (not your login password) is used.

**Feed shows no posts on Bluesky**
1. Confirm the centroid is built ("Centroid ✓" on the feed detail page)
2. Give it a few minutes for the firehose to accumulate matching posts
3. Try lowering `similarity_threshold` for niche topics

---

[← Back to Home](index.md)
