"""
End-to-end NLP pipeline.
Raw post from firehose → domain_label + stance_label + embedding → saved to DB.
"""
import datetime as dt
from langdetect import detect, LangDetectException

from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier
from nlp.stance_detector import StanceDetectorV1
from database.models import Post, db
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
