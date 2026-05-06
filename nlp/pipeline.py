"""
NLP pipelines:

  NLPPipeline  — legacy single-topic pipeline (politics/science).
                 Kept for backward compatibility and the default feed setup.

  MultiPipeline — dynamic multi-feed pipeline.
                  Loads all active Feed records from DB, builds a keyword
                  index for O(1) candidate selection, and uses FeedClassifier
                  for cosine-similarity scoring per feed.
                  Call reload() to pick up new/changed feeds without restart.
"""
import datetime as dt
import threading
import json
from langdetect import detect, LangDetectException

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier, FeedClassifier
from nlp.stance_detector import StanceDetectorV1
from nlp.model_manager import ModelManager
from database.models import Post, Feed, db
from config.settings import MIN_TURKISH_PROB


class NLPPipeline:
    """
    Orchestrates all NLP components in sequence.

    Usage:
        pipeline = NLPPipeline()
        pipeline.load_models()

        result = pipeline.process_post(
            uri="at://did:xxx/app.bsky.feed.post/xxx",
            cid="bafyxxx",
            author_did="did:plc:xxx",
            author_handle="user.bsky.social",
            text="Meclis te onemli karar alindi",
            created_at=datetime.datetime.utcnow()
        )
        # Returns a Post instance, or None if filtered out
    """

    def __init__(self):
        self.embedder = None
        self.domain_classifier = None
        self.stance_detector = None
        self._loaded = False

    def load_models(
        self,
        centroid_path: str = "data/centroids.npy",
        stance_path: str = "data/stance_centroids.npy"
    ) -> None:
        """Load all models. Call once at application startup."""
        print("Loading NLP models...")

        self.embedder = TurkishEmbedder()

        self.domain_classifier = DomainClassifier(self.embedder)
        try:
            self.domain_classifier.load_centroids(centroid_path)
        except FileNotFoundError:
            print("Centroid file not found — building from keywords...")
            self.domain_classifier.build_centroids_from_keywords()
            self.domain_classifier.save_centroids(centroid_path)

        self.stance_detector = StanceDetectorV1(self.embedder)
        try:
            self.stance_detector.load(stance_path)
        except FileNotFoundError:
            print("Stance centroid file not found — building from keywords...")
            self.stance_detector.build_stance_centroids_from_keywords()
            self.stance_detector.save(stance_path)

        self._loaded = True
        print("NLP pipeline ready.")

    def is_turkish(self, text: str) -> bool:
        """Return True if the text is detected as Turkish."""
        try:
            return detect(text) == "tr"
        except LangDetectException:
            return False

    def process_post(
        self,
        uri: str,
        cid: str,
        author_did: str,
        author_handle: str,
        text: str,
        created_at: dt.datetime
    ):
        """
        Process a raw post and save it to the database.

        Returns:
            Post instance if saved, None if filtered out
            (non-Turkish text or irrelevant domain)
        """
        assert self._loaded, "load_models() must be called before process_post()"

        # Step 1: Turkish language filter
        if not self.is_turkish(text):
            return None

        # Step 2: Compute embedding
        embedding = self.embedder.embed(text)
        embedding_json = self.embedder.vector_to_json(embedding)

        # Step 3: Domain classification
        domain_label, domain_score = self.domain_classifier.classify(text, embedding)

        if domain_label == 'other':
            return None  # Irrelevant post, skip

        # Step 4: Stance detection (politics posts only)
        stance_label = 'neutral'
        stance_score = 0.0

        if domain_label in ('politics', 'both'):
            stance_label, stance_score = self.stance_detector.detect_stance(text, embedding)

        # Step 5: Initial feed score (updated periodically by feed_logic)
        feed_score = domain_score

        # Step 6: Save to database (skip if URI already exists)
        post, created = Post.get_or_create(
            uri=uri,
            defaults={
                'cid': cid,
                'author_did': author_did,
                'author_handle': author_handle,
                'text': text,
                'domain_label': domain_label,
                'stance_label': stance_label,
                'domain_score': domain_score,
                'stance_score': stance_score,
                'embedding': embedding_json,
                'created_at': created_at,
                'language': 'tr',
                'feed_score': feed_score
            }
        )

        return post


# ---------------------------------------------------------------------------
# MultiPipeline — dynamic multi-feed
# ---------------------------------------------------------------------------

class MultiPipeline:
    """
    Processes firehose posts against all active Feed records.

    Internal state:
      keyword_index   {keyword_lower: {feed_id, ...}}  — built from DB
      classifiers     {feed_id: FeedClassifier}
      embedders       {model_type: TurkishEmbedder}
      feeds           {feed_id: Feed}

    Thread safety: reload() acquires _lock before swapping state.
    """

    RELOAD_INTERVAL = 60  # seconds between DB polls

    def __init__(self):
        self._lock = threading.Lock()
        self._feeds: dict = {}
        self._keyword_index: dict = {}
        self._classifiers: dict = {}
        self._embedders: dict = {}
        self._last_reload: dt.datetime | None = None

    # ------------------------------------------------------------------
    # Startup & reload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initial load from DB. Call once at worker startup."""
        self._do_reload()
        print(f"MultiPipeline ready: {len(self._feeds)} feeds, "
              f"{len(self._keyword_index)} unique keywords")

    def reload(self) -> int:
        """
        Reload feeds from DB if any Feed.updated_at is newer than last reload.
        Returns number of changed feeds detected.
        """
        if self._last_reload is None:
            self._do_reload()
            return len(self._feeds)

        changed = list(
            Feed.select()
            .where(Feed.is_active == True, Feed.updated_at > self._last_reload)
        )
        if changed:
            print(f"MultiPipeline: {len(changed)} feed(s) changed — reloading")
            self._do_reload()
        return len(changed)

    def _do_reload(self) -> None:
        feeds = list(Feed.select().where(Feed.is_active == True))

        new_feeds: dict = {}
        new_kw_index: dict = {}
        new_classifiers: dict = {}
        new_embedders: dict = {}

        for feed in feeds:
            new_feeds[feed.id] = feed

            # Keyword index
            for kw in feed.get_keywords():
                k = kw.strip().lower()
                if k:
                    new_kw_index.setdefault(k, set()).add(feed.id)

            # Embedder (cached via ModelManager singleton)
            mt = feed.embedding_model or "berturk"
            if mt not in new_embedders:
                new_embedders[mt] = ModelManager.get_embedder(mt)

            # Classifier (only if centroid is available)
            if feed.centroid:
                new_classifiers[feed.id] = FeedClassifier(feed, new_embedders[mt])

        # Auto-build centroids for any feeds that are missing them
        for feed in feeds:
            if not feed.centroid and feed.get_seed_sentences():
                try:
                    print(f"Building missing centroid for feed: {feed.feed_id}")
                    mt = feed.embedding_model or "berturk"
                    if mt not in new_embedders:
                        new_embedders[mt] = ModelManager.get_embedder(mt)
                    embedder = new_embedders[mt]
                    vecs = embedder.embed_batch(feed.get_seed_sentences())
                    centroid = vecs.mean(axis=0)
                    centroid /= np.linalg.norm(centroid)
                    feed.set_centroid(centroid)
                    feed.touch()
                    new_classifiers[feed.id] = FeedClassifier(feed, embedder)
                    print(f"Centroid built for: {feed.feed_id}")
                except Exception as exc:
                    print(f"Centroid build failed for {feed.feed_id}: {exc}")

        with self._lock:
            self._feeds = new_feeds
            self._keyword_index = new_kw_index
            self._classifiers = new_classifiers
            self._embedders = new_embedders
            self._last_reload = dt.datetime.now(dt.timezone.utc)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def get_candidate_feed_ids(self, text: str) -> set:
        """O(K) keyword scan — K = unique keywords across all feeds."""
        text_lower = text.lower()
        matched: set = set()
        for kw, fids in self._keyword_index.items():
            if kw in text_lower:
                matched.update(fids)
        return matched

    @staticmethod
    def _check_language(text: str, language: str) -> bool:
        if language == "multi":
            return True
        try:
            return detect(text) == language
        except LangDetectException:
            return language == "tr"

    def process_post(
        self,
        uri: str,
        cid: str,
        author_did: str,
        author_handle: str,
        text: str,
        created_at: dt.datetime,
        force_candidate_ids: set | None = None,
    ) -> "Post | None":
        """
        Classify a post against all candidate feeds and save the best match.

        Steps:
          1. Fast keyword scan → candidate feed_ids
          2. Per candidate: language check + cosine similarity
          3. Save post linked to the highest-scoring feed (if any exceed threshold)

        Returns the saved Post, or None if no feed matched.
        """
        with self._lock:
            feeds = self._feeds
            classifiers = self._classifiers
            embedders = self._embedders

        candidate_ids = force_candidate_ids if force_candidate_ids is not None \
            else self.get_candidate_feed_ids(text)

        if not candidate_ids:
            return None

        # Pre-compute one embedding per model type used by candidates
        embeddings: dict = {}  # model_type → ndarray
        best_feed_id: int | None = None
        best_score = -1.0

        for fid in candidate_ids:
            feed = feeds.get(fid)
            if not feed:
                continue

            clf = classifiers.get(fid)
            if clf is None:
                continue  # no centroid yet

            if not self._check_language(text, feed.language):
                continue

            mt = feed.embedding_model or "berturk"
            if mt not in embeddings:
                emb = embedders.get(mt)
                if emb is None:
                    continue
                embeddings[mt] = emb.embed(text)

            score, matched = clf.classify(text, embedding=embeddings[mt])
            if matched and score > best_score:
                best_score = score
                best_feed_id = fid

        if best_feed_id is None:
            return None

        feed = feeds[best_feed_id]
        mt = feed.embedding_model or "berturk"
        embedding = embeddings.get(mt)
        emb_obj = embedders.get(mt)
        embedding_json = emb_obj.vector_to_json(embedding) if (emb_obj and embedding is not None) else None

        # Legacy domain_label for backward compat
        _domain_map = {"turkiye-siyaset": "politics", "turkiye-bilim": "science"}
        domain_label = _domain_map.get(feed.feed_id, feed.feed_id)

        post, _ = Post.get_or_create(
            uri=uri,
            defaults={
                "cid": cid,
                "author_did": author_did,
                "author_handle": author_handle,
                "text": text,
                "feed_id": best_feed_id,
                "domain_label": domain_label,
                "domain_score": best_score,
                "embedding": embedding_json,
                "created_at": created_at,
                "language": feed.language,
                "feed_score": best_score,
            },
        )
        return post

    # ------------------------------------------------------------------
    # Centroid builder (called by admin after LLM seed generation)
    # ------------------------------------------------------------------

    @staticmethod
    def build_centroid_for_feed(feed: "Feed") -> np.ndarray:
        """
        Embed seed_sentences and compute the mean (normalised) centroid.
        Saves the result to feed.centroid and calls feed.touch().
        """
        sentences = feed.get_seed_sentences()
        if not sentences:
            raise ValueError(f"Feed '{feed.feed_id}' has no seed sentences")

        embedder = ModelManager.get_embedder(feed.embedding_model or "berturk")
        vecs = embedder.embed_batch(sentences)
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        feed.set_centroid(centroid)
        feed.touch()
        return centroid
