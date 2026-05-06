"""
AT Protocol firehose subscriber — multi-feed edition.

Fast-to-slow filter chain:
  1. Event type == 'app.bsky.feed.post'
  2. Author DID in any feed's seed set  OR  text contains any feed keyword  (O(1))
  3. MultiPipeline: per-feed language check + centroid similarity            (slow)

Worker thread polls the DB every RELOAD_INTERVAL seconds so new feeds
added via the admin panel are picked up without restarting the process.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import threading
import datetime as dt
from collections import deque

from atproto import CAR, FirehoseSubscribeReposClient, parse_subscribe_repos_message, models
from atproto.exceptions import FirehoseError

from database.models import db, Post, TrackedUser, Feed, create_tables
from nlp.pipeline import MultiPipeline
from config.settings import POLITICS_KEYWORDS, SCIENCE_KEYWORDS

RELOAD_INTERVAL = 60   # seconds between DB polls for feed config changes


class FirehoseProcessor:
    """
    Subscribes to the AT Protocol firehose and routes posts to active feeds.

    Architecture:
      Main thread      → receives firehose messages, applies fast filters,
                         pushes candidates to self.post_queue
      process thread   → drains queue in batches, calls MultiPipeline
      reload thread    → polls DB every RELOAD_INTERVAL for feed changes,
                         triggers pipeline.reload()
      stats thread     → prints throughput every 60s
    """

    def __init__(self):
        self.pipeline = MultiPipeline()
        self.seed_dids: set = set()
        self.post_queue: deque = deque(maxlen=1000)
        self.running = False
        self._stats = {'received': 0, 'queued': 0, 'processed': 0, 'saved': 0}
        self._keyword_snapshot: set = set()  # merged keywords for fast O(1) check

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Connect to DB, initialise default feeds, load NLP pipeline."""
        db.connect(reuse_if_open=True)
        create_tables()
        _setup_default_feeds(self.pipeline)

        # Load pipeline (also rebuilds keyword index)
        self.pipeline.load()
        self._rebuild_keyword_snapshot()

        # Load seed DIDs from all active tracked users
        self.seed_dids = set(
            u.did for u in TrackedUser.select(TrackedUser.did)
            .where(TrackedUser.is_active == True)
        )
        print(f"Seed DIDs loaded: {len(self.seed_dids)}")

    def _rebuild_keyword_snapshot(self) -> None:
        """Flatten the keyword index into a set for O(1) first-pass checks."""
        self._keyword_snapshot = set(self.pipeline._keyword_index.keys())

    # ------------------------------------------------------------------
    # Message handler (called on main thread — must be fast)
    # ------------------------------------------------------------------

    def on_message_handler(self, message) -> None:
        try:
            commit = parse_subscribe_repos_message(message)

            if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                return

            car = CAR.from_bytes(commit.blocks)

            for op in commit.ops:
                self._stats['received'] += 1

                if op.action != 'create':
                    continue

                collection, _, rkey = op.path.partition('/')
                if collection != 'app.bsky.feed.post':
                    continue

                record = car.blocks.get(op.cid)
                if not record:
                    continue

                text = record.get('text', '')
                if not text or len(text) < 10:
                    continue

                author_did = commit.repo
                uri = f"at://{author_did}/app.bsky.feed.post/{rkey}"

                try:
                    created_at = dt.datetime.fromisoformat(
                        record.get('createdAt', '').replace('Z', '+00:00')
                    )
                except (ValueError, AttributeError):
                    created_at = dt.datetime.now(dt.timezone.utc)

                is_seed = author_did in self.seed_dids
                has_keyword = self._has_relevant_keyword(text)

                if is_seed or has_keyword:
                    self.post_queue.append({
                        'uri': uri,
                        'cid': str(op.cid),
                        'author_did': author_did,
                        'author_handle': '',
                        'text': text,
                        'created_at': created_at,
                    })
                    self._stats['queued'] += 1

        except Exception:
            pass

    def _has_relevant_keyword(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._keyword_snapshot)

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _process_queue_worker(self) -> None:
        """Drain queue in batches of 32, every 5 s."""
        while self.running:
            batch = []
            while self.post_queue and len(batch) < 32:
                batch.append(self.post_queue.popleft())

            if not batch:
                time.sleep(5)
                continue

            for post_data in batch:
                try:
                    result = self.pipeline.process_post(**post_data)
                    self._stats['processed'] += 1
                    if result:
                        self._stats['saved'] += 1
                except Exception as e:
                    print(f"Post processing error: {e}")

            print(
                f"Batch done: {len(batch)} | "
                f"queued={self._stats['queued']} saved={self._stats['saved']}"
            )

    def _reload_worker(self) -> None:
        """Poll DB every RELOAD_INTERVAL for feed config changes."""
        while self.running:
            time.sleep(RELOAD_INTERVAL)
            try:
                changed = self.pipeline.reload()
                if changed:
                    self._rebuild_keyword_snapshot()
                    print(f"Feed config reloaded: {len(self.pipeline._feeds)} active feeds")
            except Exception as e:
                print(f"Reload error: {e}")

    def _stats_reporter(self) -> None:
        while self.running:
            time.sleep(60)
            print(
                f"[Stats] received={self._stats['received']} "
                f"queued={self._stats['queued']} "
                f"saved={self._stats['saved']}"
            )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self.setup()

        for target in (self._process_queue_worker, self._reload_worker, self._stats_reporter):
            threading.Thread(target=target, daemon=True).start()

        client = FirehoseSubscribeReposClient()
        print("Connecting to firehose...")

        while self.running:
            try:
                client.start(self.on_message_handler)
            except FirehoseError as e:
                print(f"Firehose error, reconnecting in 10s: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(5)


# ---------------------------------------------------------------------------
# Default-feed bootstrapper (runs once if DB is empty of Feed records)
# ---------------------------------------------------------------------------

def _setup_default_feeds(pipeline: MultiPipeline) -> None:
    """
    Create the two legacy feeds (politics / science) in the DB if none exist.
    Centroids are built from DomainClassifier seed sentences.
    """
    if Feed.select().count() > 0:
        return

    print("No feeds found — bootstrapping default feeds (politics, science)…")

    from config.settings import POLITICS_KEYWORDS, SCIENCE_KEYWORDS, FEED_URI_POLITICS, FEED_URI_SCIENCE
    from nlp.domain_classifier import DomainClassifier
    from nlp.model_manager import ModelManager
    import json, numpy as np

    embedder = ModelManager.get_embedder("berturk")
    classifier = DomainClassifier(embedder)
    classifier.build_centroids_from_keywords()

    politics = Feed.create(
        feed_id="turkiye-siyaset",
        display_name="Türkiye Siyaset",
        description="Türkiye siyasetine dair Bluesky paylaşımları.",
        language="tr",
        topic="Türkiye siyaseti",
        embedding_model="berturk",
        similarity_threshold=0.30,
        at_uri=FEED_URI_POLITICS or None,
        rkey="turkiye-siyaset",
    )
    politics.set_keywords(POLITICS_KEYWORDS)
    politics.set_seed_sentences(DomainClassifier.get_politics_sentences())
    politics.set_centroid(classifier.centroids["politics"])
    politics.save()

    science = Feed.create(
        feed_id="turkiye-bilim",
        display_name="Türkiye Bilim",
        description="Türkiye'den bilim, akademi ve araştırma paylaşımları.",
        language="tr",
        topic="Türkiye bilim ve akademi",
        embedding_model="berturk",
        similarity_threshold=0.30,
        at_uri=FEED_URI_SCIENCE or None,
        rkey="turkiye-bilim",
    )
    science.set_keywords(SCIENCE_KEYWORDS)
    science.set_seed_sentences(DomainClassifier.get_science_sentences())
    science.set_centroid(classifier.centroids["science"])
    science.save()

    print("Default feeds created: turkiye-siyaset, turkiye-bilim")


if __name__ == "__main__":
    processor = FirehoseProcessor()
    processor.start()
