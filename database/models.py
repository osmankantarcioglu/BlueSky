"""
Peewee ORM models for SQLite (development) / PostgreSQL (production).
"""
import os, sys
# Ensure project root is on the path regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model,
    CharField, TextField, FloatField, IntegerField,
    BooleanField, DateTimeField
)
from config.settings import DATABASE_PATH

# Use PostgreSQL when DATABASE_URL is set (Railway production),
# otherwise fall back to SQLite (local development).
_DATABASE_URL = os.getenv("DATABASE_URL")

if _DATABASE_URL:
    # Railway provides: postgresql://user:pass@host:port/dbname
    import urllib.parse as _urlparse
    _u = _urlparse.urlparse(_DATABASE_URL)
    db = PostgresqlDatabase(
        _u.path.lstrip("/"),
        user=_u.username,
        password=_u.password,
        host=_u.hostname,
        port=_u.port or 5432,
        autorollback=True,
    )
else:
    db = SqliteDatabase(DATABASE_PATH)


class BaseModel(Model):
    class Meta:
        database = db


class TrackedUser(BaseModel):
    """Turkish users being monitored (politicians + discovered accounts)."""
    did = CharField(primary_key=True)
    handle = CharField(index=True)
    display_name = CharField(null=True)
    party = CharField(null=True)       # AKP, CHP, MHP, etc.
    stance = CharField(null=True)      # 'alliance', 'opposition', 'unknown'
    domain = CharField(null=True)      # 'politics', 'science', 'both'
    source = CharField(null=True)      # csv, starter_pack, search
    created_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))
    is_active = BooleanField(default=True)

    class Meta:
        table_name = "tracked_users"


class Post(BaseModel):
    """
    Posts collected from the firehose and processed by the NLP pipeline.
    The embedding column stores the BERTurk vector (768-dim) as a JSON string.
    """
    uri = CharField(primary_key=True)   # at://did:xxx/app.bsky.feed.post/xxx
    cid = CharField()
    author_did = CharField(index=True)
    author_handle = CharField(null=True)
    text = TextField()

    # NLP results
    domain_label = CharField(null=True)  # 'politics', 'science', 'both', 'other'
    stance_label = CharField(null=True)  # 'alliance', 'opposition', 'neutral'
    domain_score = FloatField(null=True) # Cosine similarity score
    stance_score = FloatField(null=True) # Stance model confidence
    embedding = TextField(null=True)     # JSON string: [0.12, -0.34, ...]

    # Metadata
    created_at = DateTimeField(index=True)
    indexed_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))
    language = CharField(null=True)      # 'tr', 'en', etc.

    # Engagement (updated from firehose events)
    like_count = IntegerField(default=0)
    repost_count = IntegerField(default=0)
    reply_count = IntegerField(default=0)

    # Combined ranking score
    feed_score = FloatField(default=0.0, index=True)

    class Meta:
        table_name = "posts"
        indexes = (
            (("domain_label", "created_at"), False),
            (("stance_label", "domain_label", "created_at"), False),
        )


class LikeEvent(BaseModel):
    """Like events received from the firehose."""
    uri = CharField()          # URI of the liked post
    liker_did = CharField()
    created_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))

    class Meta:
        table_name = "like_events"


def create_tables():
    """Create all tables. Safe to call multiple times (uses IF NOT EXISTS)."""
    with db:
        db.create_tables([TrackedUser, Post, LikeEvent], safe=True)
    print("Tables created: tracked_users, posts, like_events")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_tables()
