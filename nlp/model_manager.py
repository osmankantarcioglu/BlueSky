"""
Singleton cache for embedding models.

Keeps one TurkishEmbedder instance per model type so the worker never loads
the same ~400 MB checkpoint twice, even when multiple feeds share a model.

Supported model types (config key → HuggingFace repo):
  'berturk'      → emrecan/bert-base-turkish-cased-mean-nli-stsb-tr  (768-dim)
  'minilm'       → sentence-transformers/all-MiniLM-L6-v2             (384-dim)
  'multilingual' → sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
"""
from __future__ import annotations

import threading
from config.settings import EMBEDDING_MODELS


class ModelManager:
    _models: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get_embedder(cls, model_type: str = "berturk"):
        """Return (and cache) a TurkishEmbedder for the given model type."""
        with cls._lock:
            if model_type not in cls._models:
                cls._models[model_type] = cls._load(model_type)
        return cls._models[model_type]

    @classmethod
    def _load(cls, model_type: str):
        # Import here to avoid circular imports at module level
        from nlp.embedder import TurkishEmbedder

        model_name = EMBEDDING_MODELS.get(model_type)
        if not model_name:
            raise ValueError(
                f"Unknown embedding model type: {model_type!r}. "
                f"Valid options: {list(EMBEDDING_MODELS)}"
            )
        return TurkishEmbedder(model_name=model_name)

    @classmethod
    def preload(cls, model_types: list[str]) -> None:
        """Eagerly load a list of model types (call at startup to avoid lazy-load lag)."""
        for mt in model_types:
            cls.get_embedder(mt)

    @classmethod
    def loaded_types(cls) -> list[str]:
        return list(cls._models.keys())
