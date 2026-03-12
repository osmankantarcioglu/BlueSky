"""
AT Protocol firehose subscriber.

The firehose streams ALL public Bluesky events in real time (~200-500/sec).
Most are irrelevant — we filter fast-to-slow before hitting the NLP pipeline:

  1. Event type == 'app.bsky.feed.post' (post creation only)
  2. Author DID in seed_dids set          (O(1) lookup)
  3. Turkish keyword present in text      (regex, very fast)
  4. NLP pipeline                         (slow, only for candidates above)
"""
import os, sys
# Ensure project root is on the path regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import time
import threading
import datetime as dt
from collections import deque

from atproto import CAR, FirehoseSubscribeReposClient, parse_subscribe_repos_message, models
from atproto.exceptions import FirehoseError

from database.models import db, Post, TrackedUser
from nlp.pipeline import NLPPipeline
from config.settings import POLITICS_KEYWORDS, SCIENCE_KEYWORDS

ALL_KEYWORDS = set(POLITICS_KEYWORDS + SCIENCE_KEYWORDS)


class FirehoseProcessor:
    """
    Main class that subscribes to the firehose and processes matching posts.

    Performance notes:
    - NLP pipeline startup takes 30-60 seconds
    - Firehose delivers ~300 events/sec
    - BERTurk embedding: ~50ms/post (CPU), ~5ms/post (GPU)
    - Posts are batched (up to 32) before NLP processing
    """

    def __init__(self):
        self.nlp_pipeline = NLPPipeline()
        self.seed_dids: set = set()
        self.post_queue = deque(maxlen=1000)
        self.running = False
        self._stats = {'received': 0, 'queued': 0, 'processed': 0, 'saved': 0}

    def setup(self):
        """Load seed DIDs into memory and initialize the NLP pipeline."""
        db.connect(reuse_if_open=True)

        self.seed_dids = set(
            u.did for u in TrackedUser.select(TrackedUser.did).where(
                TrackedUser.is_active == True
            )
        )
        print(f"Seed DIDs loaded: {len(self.seed_dids)}")

        self.nlp_pipeline.load_models()

    def _has_relevant_keyword(self, text: str) -> bool:
        """Fast keyword pre-filter before expensive NLP."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in ALL_KEYWORDS)

    def on_message_handler(self, message) -> None:
        """
        Called for every firehose message.
        Decodes the CAR block, applies fast filters, pushes candidates to queue.

        In atproto SDK, commit ops are objects (not dicts):
          op.action  → 'create' | 'update' | 'delete'
          op.path    → 'app.bsky.feed.post/rkey'
          op.cid     → CID of the record block
        Records are stored in the CAR blocks attached to the commit.
        """
        try:
            commit = parse_subscribe_repos_message(message)

            if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                return

            car = CAR.from_bytes(commit.blocks)

            for op in commit.ops:
                self._stats['received'] += 1

                if op.action != 'create':
                    continue

                # op.path = "collection/rkey", e.g. "app.bsky.feed.post/3abc"
                collection, _, rkey = op.path.partition('/')
                if collection != 'app.bsky.feed.post':
                    continue

                # Decode record from CAR blocks
                record = car.blocks.get(op.cid)
                if not record:
                    continue

                text = record.get('text', '')
                if not text or len(text) < 10:
                    continue

                author_did = commit.repo
                uri = f"at://{author_did}/app.bsky.feed.post/{rkey}"

                created_at_str = record.get('createdAt', '')
                try:
                    created_at = dt.datetime.fromisoformat(
                        created_at_str.replace('Z', '+00:00')
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
                        'created_at': created_at
                    })
                    self._stats['queued'] += 1

        except Exception:
            pass

    def _process_queue_worker(self):
        """
        Background thread: drains the queue and runs the NLP pipeline.
        Processes in batches of up to 32 posts, or every 5 seconds.
        """
        while self.running:
            batch = []
            while self.post_queue and len(batch) < 32:
                batch.append(self.post_queue.popleft())

            if not batch:
                time.sleep(5)
                continue

            for post_data in batch:
                try:
                    result = self.nlp_pipeline.process_post(**post_data)
                    self._stats['processed'] += 1
                    if result:
                        self._stats['saved'] += 1
                except Exception as e:
                    print(f"Post processing error: {e}")

            print(
                f"Batch done: {len(batch)} processed | "
                f"total queued={self._stats['queued']} "
                f"saved={self._stats['saved']}"
            )

    def _stats_reporter(self):
        """Print throughput stats every 60 seconds."""
        while self.running:
            time.sleep(60)
            print(
                f"[Stats] received={self._stats['received']} "
                f"queued={self._stats['queued']} "
                f"saved={self._stats['saved']}"
            )

    def start(self):
        """Start firehose subscription and background workers."""
        self.running = True
        self.setup()

        worker = threading.Thread(target=self._process_queue_worker, daemon=True)
        worker.start()

        reporter = threading.Thread(target=self._stats_reporter, daemon=True)
        reporter.start()

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


if __name__ == "__main__":
    processor = FirehoseProcessor()
    processor.start()
