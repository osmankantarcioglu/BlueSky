"""
Feed ranking logic: selects and orders posts for the politics and science feeds.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
from typing import Optional
from database.models import db, Post
from config.settings import MAX_FEED_POSTS


def _recency_boost(created_at: dt.datetime, now: dt.datetime) -> float:
    """Return a recency multiplier in range [0.5, 1.0].

    Posts older than 48 hours get 0.5, fresh posts get 1.0.
    """
    age_hours = (now - created_at.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
    age_hours = max(age_hours, 0)
    return max(0.5, 1.0 - age_hours / 96)


def _engagement_score(post: Post) -> float:
    """Weighted engagement metric."""
    return post.like_count * 1.0 + post.repost_count * 2.0 + post.reply_count * 0.5


def compute_feed_score(post: Post, now: dt.datetime) -> float:
    """Combine NLP similarity score, engagement, and recency into a single rank."""
    nlp = post.domain_score or 0.0
    eng = _engagement_score(post)
    rec = _recency_boost(post.created_at, now)
    # Normalise engagement (log scale to avoid viral posts dominating)
    import math
    eng_norm = math.log1p(eng) / 10.0
    return (nlp * 0.5 + eng_norm * 0.3) * rec


def get_feed_posts(
    domain: str,
    cursor: Optional[str] = None,
    limit: int = 30,
) -> tuple[list[Post], Optional[str]]:
    """Return ranked posts for a domain feed, with optional cursor pagination.

    Args:
        domain: 'politics' or 'science'
        cursor: opaque string (ISO datetime) returned from a previous call
        limit:  number of posts to return (max 100)

    Returns:
        (posts, next_cursor) — next_cursor is None when there are no more results.
    """
    limit = min(limit, 100)
    now = dt.datetime.now(dt.timezone.utc)

    # Select posts that belong to this domain (includes 'both')
    query = (
        Post.select()
        .where(
            (Post.domain_label == domain) | (Post.domain_label == "both")
        )
        .order_by(Post.feed_score.desc(), Post.created_at.desc())
        .limit(MAX_FEED_POSTS)
    )

    # Apply cursor: skip posts with feed_score >= cursor_score (pagination)
    # We use indexed_at as the cursor value for simplicity
    if cursor:
        try:
            cursor_dt = dt.datetime.fromisoformat(cursor)
            query = query.where(Post.indexed_at < cursor_dt)
        except ValueError:
            pass  # ignore malformed cursor

    posts = list(query.limit(limit))

    # Recompute live scores and sort (scores may have changed since last save)
    posts.sort(key=lambda p: compute_feed_score(p, now), reverse=True)

    next_cursor = None
    if len(posts) == limit:
        last = posts[-1]
        next_cursor = last.indexed_at.isoformat()

    return posts, next_cursor


def refresh_feed_scores() -> int:
    """Recompute feed_score for all stored posts and persist to DB.

    Call this periodically (e.g. every 5 minutes) so the feed order
    reflects the latest engagement counts.
    """
    now = dt.datetime.now(dt.timezone.utc)
    updated = 0
    for post in Post.select():
        new_score = compute_feed_score(post, now)
        if abs(new_score - post.feed_score) > 0.001:
            post.feed_score = new_score
            post.save(only=[Post.feed_score])
            updated += 1
    return updated
