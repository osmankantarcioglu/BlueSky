"""
Peewee ORM models for SQLite (development) / PostgreSQL (production).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model,
    CharField, TextField, FloatField, IntegerField,
    BooleanField, DateTimeField, AutoField, ForeignKeyField,
)
from config.settings import DATABASE_PATH

_DATABASE_URL = os.getenv("DATABASE_URL")

if _DATABASE_URL:
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


class Feed(BaseModel):
    """Dynamically created feed topic (e.g. 'turkiye-siyaset', 'world-cup')."""
    id = AutoField()
    feed_id = CharField(unique=True, max_length=64)    # URL-safe slug
    display_name = CharField(max_length=128)
    description = TextField(default='')
    language = CharField(default='tr')                 # 'tr' | 'en' | 'multi'
    topic = CharField(max_length=256)

    # Bluesky registration (set after publish_feed is called)
    at_uri = CharField(null=True)
    rkey = CharField(null=True)

    # NLP config stored as JSON text
    keywords = TextField(default='[]')         # JSON list[str]
    seed_sentences = TextField(default='[]')   # JSON list[str]
    centroid = TextField(null=True)            # JSON list[float]  (normalized 768-dim)

    # Which sentence-transformer to use
    embedding_model = CharField(default='berturk')  # 'berturk' | 'minilm' | 'multilingual'
    similarity_threshold = FloatField(default=0.30)

    is_active = BooleanField(default=True)
    created_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))

    class Meta:
        table_name = 'feeds'

    # --- helpers ---

    def get_keywords(self) -> list:
        return json.loads(self.keywords) if self.keywords else []

    def set_keywords(self, lst: list) -> None:
        self.keywords = json.dumps(lst, ensure_ascii=False)

    def get_seed_sentences(self) -> list:
        return json.loads(self.seed_sentences) if self.seed_sentences else []

    def set_seed_sentences(self, lst: list) -> None:
        self.seed_sentences = json.dumps(lst, ensure_ascii=False)

    def get_centroid(self):
        import numpy as np
        return np.array(json.loads(self.centroid), dtype=np.float32) if self.centroid else None

    def set_centroid(self, vector) -> None:
        arr = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
        self.centroid = json.dumps(arr)

    def touch(self) -> None:
        self.updated_at = dt.datetime.now(dt.timezone.utc)
        self.save(only=[Feed.updated_at])


class TrackedUser(BaseModel):
    """Users being monitored (politicians, academics, etc.)."""
    did = CharField(primary_key=True)
    handle = CharField(index=True)
    display_name = CharField(null=True)
    party = CharField(null=True)
    stance = CharField(null=True)      # 'alliance' | 'opposition' | 'unknown'
    domain = CharField(null=True)      # 'politics' | 'science' | 'both'
    source = CharField(null=True)      # 'csv' | 'starter_pack' | 'search'
    created_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))
    is_active = BooleanField(default=True)

    class Meta:
        table_name = "tracked_users"


class FeedSeedUser(BaseModel):
    """Per-feed seed user list (Feed ↔ TrackedUser many-to-many)."""
    feed = ForeignKeyField(Feed, backref='seed_user_links', on_delete='CASCADE')
    user_did = CharField()

    class Meta:
        table_name = 'feed_seed_users'
        # Unique constraint is created via raw SQL in _run_migrations()
        # to avoid Peewee's ForeignKeyField column-name resolution ambiguity.


class Post(BaseModel):
    """
    Posts collected from the firehose and processed by the NLP pipeline.

    feed_id (FK → Feed) is the primary classification; domain_label is kept
    for backward compatibility with data collected before the dynamic-feed
    migration.
    """
    uri = CharField(primary_key=True)
    cid = CharField()
    author_did = CharField(index=True)
    author_handle = CharField(null=True)
    text = TextField()

    # Dynamic feed reference (nullable for legacy rows)
    feed = ForeignKeyField(Feed, null=True, backref='posts', on_delete='SET NULL')

    # Legacy columns (kept for backward compat)
    domain_label = CharField(null=True)   # 'politics' | 'science'
    stance_label = CharField(null=True)   # 'alliance' | 'opposition' | 'neutral'
    domain_score = FloatField(null=True)
    stance_score = FloatField(null=True)
    embedding = TextField(null=True)      # JSON float list

    created_at = DateTimeField(index=True)
    indexed_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))
    language = CharField(null=True)

    like_count = IntegerField(default=0)
    repost_count = IntegerField(default=0)
    reply_count = IntegerField(default=0)

    feed_score = FloatField(default=0.0, index=True)

    class Meta:
        table_name = "posts"
        indexes = (
            (("domain_label", "created_at"), False),
            (("stance_label", "domain_label", "created_at"), False),
        )


class LikeEvent(BaseModel):
    uri = CharField()
    liker_did = CharField()
    created_at = DateTimeField(default=lambda: dt.datetime.now(dt.timezone.utc))

    class Meta:
        table_name = "like_events"


ALL_TABLES = [Feed, TrackedUser, FeedSeedUser, Post, LikeEvent]


def create_tables() -> None:
    """
    Create all tables independently. Each model gets its own savepoint so a
    failure on one table (e.g. posts FK index on missing feed_id column) does
    NOT abort the whole PostgreSQL transaction and block later tables.
    """
    if db.is_closed():
        db.connect(reuse_if_open=True)

    for model in ALL_TABLES:
        try:
            with db.atomic():
                model.create_table(safe=True)
        except Exception as exc:
            print(f"create_table note ({model._meta.table_name}): {exc}")

    _run_migrations()
    print("DB tables ready.")


def _run_migrations() -> None:
    """Incremental schema changes applied after table creation."""

    def _sql(statement: str) -> None:
        try:
            with db.atomic():
                db.execute_sql(statement)
        except Exception as exc:
            print(f"Migration skipped: {exc}")

    if isinstance(db, SqliteDatabase):
        cols = [r[1] for r in db.execute_sql("PRAGMA table_info(posts)").fetchall()]
        if 'feed_id' not in cols:
            _sql("ALTER TABLE posts ADD COLUMN feed_id INTEGER REFERENCES feeds(id)")
            print("Migration applied: posts.feed_id (SQLite)")
        _sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedseeduser_uniq "
            "ON feed_seed_users(feed_id, user_did)"
        )
    else:
        _sql(
            "ALTER TABLE posts ADD COLUMN IF NOT EXISTS "
            "feed_id INTEGER REFERENCES feeds(id) ON DELETE SET NULL"
        )
        _sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedseeduser_uniq "
            "ON feed_seed_users(feed_id, user_did)"
        )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_tables()
