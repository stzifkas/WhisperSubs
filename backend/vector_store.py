"""Per-session in-memory vector store for semantic transcript retrieval.

Used by the chat endpoint to find contextually relevant transcript segments
regardless of when they occurred in the session (vs. recency-only window).
"""
import logging

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from . import config

logger = logging.getLogger(__name__)

_embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    api_key=config.OPENAI_API_KEY,
)

_stores: dict[str, InMemoryVectorStore] = {}


def _get_or_create(session_id: str) -> InMemoryVectorStore:
    if session_id not in _stores:
        _stores[session_id] = InMemoryVectorStore(embedding=_embeddings)
    return _stores[session_id]


async def add_segment(session_id: str, text: str, metadata: dict) -> None:
    """Embed and store a refined transcript segment. Call via asyncio.create_task()."""
    try:
        store = _get_or_create(session_id)
        await store.aadd_texts([text], metadatas=[metadata])
    except Exception as exc:
        logger.warning("add_segment failed for session %s: %s", session_id, exc)


async def search_segments(session_id: str, query: str, k: int = 6) -> list[str]:
    """Return up to k transcript segments most semantically similar to query."""
    if session_id not in _stores:
        return []
    try:
        store = _stores[session_id]
        docs = await store.asimilarity_search(query, k=k)
        return [d.page_content for d in docs]
    except Exception as exc:
        logger.warning("search_segments failed for session %s: %s", session_id, exc)
        return []


def remove_store(session_id: str) -> None:
    """Free the vector store for a completed session."""
    _stores.pop(session_id, None)
