"""
BERTurk-based sentence embedding module.
Model: emrecan/bert-base-turkish-cased-mean-nli-stsb-tr

Alternative models (test for better Turkish performance):
- "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  (multilingual, fast)
- "dbmdz/bert-base-turkish-cased"                                (base BERTurk)
- "dbmdz/bert-base-turkish-128k-cased"                          (larger vocabulary)
"""
import re
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


class TurkishEmbedder:
    """
    Sentence Transformer wrapper optimized for Turkish text.

    Usage:
        embedder = TurkishEmbedder()
        vector = embedder.embed("Meclis'te yeni kanun teklifi kabul edildi.")
        # vector.shape == (768,)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Load the embedding model.

        Note: First run downloads ~400MB. Subsequent runs load from cache.
        GPU is used automatically if available, otherwise falls back to CPU.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} ({device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print("Model loaded.")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text into a vector.
        Returns: numpy array, shape (768,), dtype float32
        """
        text = self._preprocess(text)
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector

    def embed_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts in batches (use this for firehose processing).
        Returns: numpy array, shape (len(texts), 768)

        batch_size can be increased to 64 or 128 if GPU memory allows.
        """
        texts = [self._preprocess(t) for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return vectors

    def _preprocess(self, text: str) -> str:
        """
        Clean post text before embedding.
        - Remove URLs (noise, no semantic value)
        - Remove @mentions
        - Remove # from hashtags but keep the word
        - Collapse extra whitespace
        - Truncate to 512 chars (safe limit for BERT token budget)
        """
        text = re.sub(r'http\S+', '', text)        # Remove URLs
        text = re.sub(r'@\w+', '', text)            # Remove mentions
        text = re.sub(r'#(\w+)', r'\1', text)       # Strip # but keep word
        text = ' '.join(text.split())               # Collapse whitespace
        return text[:512]

    def vector_to_json(self, vector: np.ndarray) -> str:
        """Serialize vector to JSON string for database storage."""
        return json.dumps(vector.tolist())

    def json_to_vector(self, json_str: str) -> np.ndarray:
        """Deserialize JSON string from database back to numpy array."""
        return np.array(json.loads(json_str), dtype=np.float32)
