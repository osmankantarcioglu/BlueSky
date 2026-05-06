"""
Feed ranking logic: selects and orders posts for any active feed.

Two public entry points:
  get_feed_posts(domain, cursor, limit)      — legacy, queries by domain_label
  get_posts_for_feed(feed_id, cursor, limit) — new,    queries by Feed FK
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
import math
from typing import Optional

from database.models import db, Post, Feed
from config.settings import MAX_FEED_POSTS


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _recency_boost(created_at: dt.datetime, now: dt.datetime) -> float:
    """Linear boost in [0.5, 1.0] — posts older than 48 h get 0.5."""
    age_h = (now - created_at.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
    return max(0.5, 1.0 - max(age_h, 0) / 96)


def _engagement_score(post: Post) -> float:
    return post.like_count * 1.0 + post.repost_count * 2.0 + post.reply_count * 0.5


def compute_feed_score(post: Post, now: dt.datetime) -> float:
    nlp = post.domain_score or 0.0
    eng_norm = math.log1p(_engagement_score(post)) / 10.0
    rec = _recency_boost(post.created_at, now)
    return (nlp * 0.5 + eng_norm * 0.3) * rec


# ---------------------------------------------------------------------------
# Legacy query (domain_label-based — for backward compat)
# ---------------------------------------------------------------------------

def get_feed_posts(
    domain: str,
    cursor: Optional[str] = None,
    limit: int = 30,
) -> tuple[list, Optional[str]]:
    """Return ranked posts for a legacy domain label ('politics' / 'science')."""
    limit = min(limit, 100)
    now = dt.datetime.now(dt.timezone.utc)

    query = (
        Post.select()
        .where(Post.domain_label == domain)
        .order_by(Post.feed_score.desc(), Post.created_at.desc())
        .limit(MAX_FEED_POSTS)
    )

    if cursor:
        try:
            query = query.where(Post.indexed_at < dt.datetime.fromisoformat(cursor))
        except ValueError:
            pass

    posts = list(query.limit(limit))
    posts.sort(key=lambda p: compute_feed_score(p, now), reverse=True)

    next_cursor = posts[-1].indexed_at.isoformat() if len(posts) == limit else None
    return posts, next_cursor


# ---------------------------------------------------------------------------
# New query (Feed FK-based)
# ---------------------------------------------------------------------------

def get_posts_for_feed(
    feed_id: int,
    cursor: Optional[str] = None,
    limit: int = 30,
) -> tuple[list, Optional[str]]:
    """
    Return ranked posts for a Feed record.

    Falls back to domain_label lookup for rows that pre-date the migration
    (i.e. feed_id IS NULL but domain_label matches the feed's slug).
    """
    limit = min(limit, 100)
    now = dt.datetime.now(dt.timezone.utc)

    feed = Feed.get_or_none(Feed.id == feed_id)
    if feed is None:
        return [], None

    # Posts saved by the new multi-pipeline have feed_id set.
    # Legacy rows (before migration) have feed_id=NULL but domain_label set.
    _domain_map = {"turkiye-siyaset": "politics", "turkiye-bilim": "science"}
    legacy_label = _domain_map.get(feed.feed_id)

    if legacy_label:
        query = Post.select().where(
            (Post.feed_id == feed_id) | (Post.domain_label == legacy_label)
        )
    else:
        query = Post.select().where(Post.feed_id == feed_id)

    query = query.order_by(Post.feed_score.desc(), Post.created_at.desc()).limit(MAX_FEED_POSTS)

    if cursor:
        try:
            query = query.where(Post.indexed_at < dt.datetime.fromisoformat(cursor))
        except ValueError:
            pass

    posts = list(query.limit(limit))
    posts.sort(key=lambda p: compute_feed_score(p, now), reverse=True)

    next_cursor = posts[-1].indexed_at.isoformat() if len(posts) == limit else None
    return posts, next_cursor


# ---------------------------------------------------------------------------
# Periodic score refresh
# ---------------------------------------------------------------------------

def refresh_feed_scores() -> int:
    """Recompute feed_score for all posts. Call periodically (e.g. every 5 min)."""
    now = dt.datetime.now(dt.timezone.utc)
    updated = 0
    for post in Post.select():
        new_score = compute_feed_score(post, now)
        if abs(new_score - post.feed_score) > 0.001:
            post.feed_score = new_score
            post.save(only=[Post.feed_score])
            updated += 1
    return updated
