"""
embeddings.py — compute sentence-transformer embeddings for chunks,
then aggregate to file-, run-, and agent-level vectors.

Falls back gracefully to TF-IDF vectors if sentence-transformers
is unavailable or the model cannot be loaded.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunking import Chunk

logger = logging.getLogger(__name__)

SEED = 42


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------

class _EmbeddingBackend:
    name: str

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError


class _SentenceTransformerBackend(_EmbeddingBackend):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"
        logger.info("Loaded sentence-transformer model: %s", model_name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vecs, dtype=np.float32)


class _TFIDFBackend(_EmbeddingBackend):
    """Fallback: TF-IDF + SVD to produce dense vectors."""

    def __init__(self, n_components: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self._normalize = normalize
        self._fitted = False
        self._corpus_cache: List[str] = []
        self.name = "tfidf+svd (fallback)"
        self.n_components = n_components

    def fit(self, texts: List[str]) -> None:
        if not texts:
            return
        tfidf = self._vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        self._svd.n_components = n_comp
        self._svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_components), dtype=np.float32)
        if not self._fitted:
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf)
        return self._normalize(vecs).astype(np.float32)


def load_backend(model_name: str, force_fallback: bool = False) -> _EmbeddingBackend:
    if force_fallback:
        logger.info("Using TF-IDF fallback backend (forced).")
        return _TFIDFBackend()
    try:
        return _SentenceTransformerBackend(model_name)
    except Exception as exc:
        logger.warning(
            "Could not load sentence-transformers model '%s': %s. "
            "Falling back to TF-IDF embeddings.",
            model_name, exc,
        )
        return _TFIDFBackend()


# ---------------------------------------------------------------------------
# Chunk → file → run → agent embeddings
# ---------------------------------------------------------------------------

def embed_chunks(
    chunk_map: Dict[str, List[Chunk]],
    backend: _EmbeddingBackend,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for every chunk.

    Returns
    -------
    dict: chunk_id → embedding vector (1-D float32 array)
    """
    all_ids: List[str] = []
    all_texts: List[str] = []

    for file_id, chunks in chunk_map.items():
        for ch in chunks:
            all_ids.append(ch.chunk_id)
            all_texts.append(ch.text)

    if not all_texts:
        logger.warning("No chunk texts to embed.")
        return {}

    # Fit TF-IDF on entire corpus if using fallback
    if isinstance(backend, _TFIDFBackend) and not backend._fitted:
        backend.fit(all_texts)

    logger.info("Embedding %d chunks with %s …", len(all_texts), backend.name)
    vecs = backend.encode(all_texts, batch_size=batch_size)

    return {cid: vecs[i] for i, cid in enumerate(all_ids)}


def aggregate_file_embeddings(
    chunk_map: Dict[str, List[Chunk]],
    chunk_embeddings: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Mean-pool chunk embeddings → file embedding.

    Returns
    -------
    dict: file_id → file embedding vector
    """
    file_embs: Dict[str, np.ndarray] = {}
    for file_id, chunks in chunk_map.items():
        vecs = [chunk_embeddings[ch.chunk_id]
                for ch in chunks
                if ch.chunk_id in chunk_embeddings]
        if vecs:
            file_embs[file_id] = np.mean(vecs, axis=0).astype(np.float32)
        else:
            logger.debug("No chunk embeddings for file %s; skipping.", file_id)
    return file_embs


def aggregate_run_embeddings(
    file_embs: Dict[str, np.ndarray],
    file_meta: List[dict],  # list of {file_id, agent, run}
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Mean-pool file embeddings → run embedding.

    Returns
    -------
    dict: (agent, run) → run embedding vector
    """
    run_vecs: Dict[Tuple[str, int], List[np.ndarray]] = {}
    for meta in file_meta:
        fid = meta["file_id"]
        key = (meta["agent"], meta["run"])
        if fid in file_embs:
            run_vecs.setdefault(key, []).append(file_embs[fid])

    return {
        k: np.mean(vs, axis=0).astype(np.float32)
        for k, vs in run_vecs.items()
        if vs
    }


def aggregate_agent_embeddings(
    run_embs: Dict[Tuple[str, int], np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Mean-pool run embeddings → agent centroid.

    Returns
    -------
    dict: agent → centroid vector
    """
    agent_vecs: Dict[str, List[np.ndarray]] = {}
    for (agent, run), vec in run_embs.items():
        agent_vecs.setdefault(agent, []).append(vec)

    return {
        agent: np.mean(vs, axis=0).astype(np.float32)
        for agent, vs in agent_vecs.items()
        if vs
    }


# ---------------------------------------------------------------------------
# Convenience matrix builders
# ---------------------------------------------------------------------------

def build_run_matrix(
    run_embs: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Return (matrix N×D, ordered_keys) for all run embeddings."""
    keys = sorted(run_embs.keys())
    if not keys:
        return np.empty((0, 0)), []
    mat = np.stack([run_embs[k] for k in keys], axis=0)
    return mat.astype(np.float32), keys
