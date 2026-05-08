"""
Business logic for feed creation.

create_feed() orchestrates:
  1. LLM seed generation
  2. Centroid computation (embedding model)
  3. DB persistence (Feed record)
  4. Bluesky registration (optional — skipped if AT URI already exists)

This module is imported by both admin/routes.py (web) and the CLI script.
"""
import json
import re
import datetime as dt

from database.models import Feed
from services.llm_seed_generator import generate_seeds


RKEY_MAX_LEN = 15  # Bluesky AT Protocol hard limit


def slugify(text: str) -> str:
    """Convert a topic string to a URL-safe feed_id (max 15 chars for Bluesky rkey)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:RKEY_MAX_LEN].strip("-")


def create_feed(
    *,
    feed_id: str,
    display_name: str,
    description: str,
    language: str,
    topic: str,
    embedding_model: str = "berturk",
    similarity_threshold: float = 0.30,
    keywords: list | None = None,
    seed_sentences: list | None = None,
    publish_to_bluesky: bool = True,
) -> Feed:
    """
    Create and persist a new Feed record.

    If keywords/seed_sentences are omitted, they are generated via LLM.
    Centroid is always computed here (requires NLP models — worker only).
    """
    if len(feed_id) > RKEY_MAX_LEN:
        raise ValueError(
            f"feed_id '{feed_id}' is {len(feed_id)} characters — "
            f"Bluesky rkey limit is {RKEY_MAX_LEN}."
        )
    if Feed.get_or_none(Feed.feed_id == feed_id):
        raise ValueError(f"Feed with id '{feed_id}' already exists")

    # 1. Generate seeds via LLM if not provided
    if not keywords or not seed_sentences:
        print(f"Generating seeds for '{topic}' via LLM…")
        data = generate_seeds(topic=topic, language=language)
        keywords = keywords or data["keywords"]
        seed_sentences = seed_sentences or data["seed_sentences"]

    # 2. Save feed record (centroid built here if NLP is available,
    #    otherwise the worker will build it on next reload cycle).
    feed_record = Feed(
        feed_id=feed_id,
        display_name=display_name,
        description=description,
        language=language,
        topic=topic,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        is_active=False,  # keep inactive until fully set up
    )
    feed_record.set_keywords(keywords)
    feed_record.set_seed_sentences(seed_sentences)
    feed_record.save(force_insert=True)

    try:
        from nlp.pipeline import MultiPipeline
        print("Building centroid from seed sentences…")
        MultiPipeline.build_centroid_for_feed(feed_record)
    except (ImportError, ModuleNotFoundError):
        print("NLP dependencies not available on this service — "
              "centroid will be built by the worker on next reload.")

    # 3. Publish to Bluesky
    if publish_to_bluesky:
        try:
            at_uri = _publish_bluesky(feed_id, display_name, description)
            feed_record.at_uri = at_uri
            feed_record.rkey = feed_id
        except Exception as exc:
            print(f"Bluesky registration skipped: {exc}")

    # 4. Activate
    feed_record.is_active = True
    feed_record.updated_at = dt.datetime.now(dt.timezone.utc)
    feed_record.save()

    print(f"Feed '{feed_id}' created successfully.")
    return feed_record


def _publish_bluesky(rkey: str, display_name: str, description: str) -> str:
    """Register a feed generator record on Bluesky. Returns the feed AT URI."""
    from atproto import Client, models as bsky_models
    from config.settings import BSKY_HANDLE, BSKY_APP_PASSWORD, FEED_DOMAIN

    if not BSKY_HANDLE or not BSKY_APP_PASSWORD:
        raise EnvironmentError("BSKY_HANDLE / BSKY_APP_PASSWORD not configured")

    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)

    feed_record = bsky_models.AppBskyFeedGenerator.Record(
        did=f"did:web:{FEED_DOMAIN}",
        display_name=display_name,
        description=description,
        created_at=client.get_current_time_iso(),
    )

    client.com.atproto.repo.put_record(
        bsky_models.ComAtprotoRepoPutRecord.Data(
            repo=client.me.did,
            collection="app.bsky.feed.generator",
            rkey=rkey,
            record=feed_record,
        )
    )

    return f"at://{client.me.did}/app.bsky.feed.generator/{rkey}"


def generate_seeds_preview(topic: str, language: str) -> dict:
    """
    Called by the admin AJAX endpoint — returns LLM output without saving.
    Returns {'keywords': [...], 'seed_sentences': [...], 'suggested_id': '...'}
    """
    data = generate_seeds(topic=topic, language=language)
    data["suggested_id"] = slugify(topic)
    return data
