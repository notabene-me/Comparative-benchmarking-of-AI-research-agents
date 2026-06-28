"""
chunking.py — split text into overlapping chunks.

Default: chunk_size=800 tokens, overlap=150 tokens.
Tokenisation: whitespace splitting (simple, fast, reproducible).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

CHUNK_SIZE   = 800   # tokens
CHUNK_OVERLAP = 150  # tokens


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    start_token: int
    end_token: int


def _tokenise(text: str) -> List[str]:
    """Whitespace tokenisation — split on any whitespace run."""
    return re.split(r"\s+", text.strip())


def chunk_text(
    text: str,
    file_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* tokens.

    Returns a list of Chunk objects including chunk text and token positions.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenise(text)
    n = len(tokens)

    if n == 0:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[Chunk] = []

    idx = 0
    chunk_num = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        chunk_tokens = tokens[idx:end]
        chunk_text_str = " ".join(chunk_tokens)
        chunks.append(
            Chunk(
                chunk_id=f"{file_id}_chunk{chunk_num:04d}",
                text=chunk_text_str,
                token_count=len(chunk_tokens),
                start_token=idx,
                end_token=end,
            )
        )
        chunk_num += 1
        if end == n:
            break
        idx += step

    return chunks


def chunk_records(
    file_texts: dict,  # {file_id: str}
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Chunk all files.

    Parameters
    ----------
    file_texts : dict mapping file_id → raw text string

    Returns
    -------
    dict mapping file_id → list[Chunk]
    """
    result = {}
    for file_id, text in file_texts.items():
        chunks = chunk_text(text, file_id, chunk_size, overlap)
        result[file_id] = chunks
        logger.debug("Chunked %s → %d chunks", file_id, len(chunks))
    return result
