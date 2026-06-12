---
layout: default
title: Home
---

# BlueSky Feed Studio

A dynamic, AI-powered Bluesky custom feed generation platform. Create, publish, and manage any number of topic-based feeds directly from a web admin panel — no code changes required.

Built on the AT Protocol firehose, NLP sentence embeddings, GPT-4o seed generation, and Vue.js — deployed on [Railway](https://railway.app).

**Live admin panel:** [web-production-77bc8f.up.railway.app/admin](https://web-production-77bc8f.up.railway.app/admin)

---

## 🎥 Demo

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px; margin: 1.5em 0;">
  <iframe src="https://www.youtube.com/embed/8HSX0YmVEkY"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0; border-radius: 8px;"
          title="BlueSky Feed Studio demo"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen>
  </iframe>
</div>

---

## ✨ What it does

- **Generates seed data with AI** — enter a topic, GPT-4o produces 100+ keywords and 50+ social-media-style sentences
- **Builds NLP centroids automatically** — the worker computes a 768-dim embedding centroid from the seed sentences
- **Publishes to Bluesky via AT Protocol** — the feed is registered under your account with one click
- **Filters the firehose in real time** — posts are classified by cosine similarity to the feed's centroid
- **Hot-reloads without restart** — the worker polls the database every 60 seconds and picks up new feeds automatically

## Default Feeds

| Feed | Rkey | Language | Description |
|------|------|----------|-------------|
| Türkiye Siyaset | `turkiye-siyaset` | Turkish | Turkish political posts — parliament, parties, elections |
| Türkiye Bilim | `turkiye-bilim` | Turkish | Turkish science — research, academia, publications |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Protocol | AT Protocol (`atproto`) |
| NLP embeddings | BERTurk · all-MiniLM-L6-v2 · paraphrase-multilingual-MiniLM |
| ML framework | PyTorch (CPU) + sentence-transformers + scikit-learn |
| AI seed generation | OpenAI GPT-4o |
| Web framework | Flask + Waitress |
| Frontend | Vue 3 (CDN) + custom glassmorphism CSS |
| Deployment | Railway (web + worker services) |

---

## Get Started

- 📦 [Installation Guide](installation.md) — set up the project locally
- 🚀 [Usage Guide](usage.md) — create and manage feeds from the admin panel

---

[View source on GitHub](https://github.com/osmankantarcioglu/BlueSky)
