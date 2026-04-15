"""
RAG Strategy Memory — ChromaDB-backed retrieval for institutional trading strategies.

Stores distilled strategies (SMC, Wyckoff, etc.) as embeddings and retrieves
the best-matching strategy given the current market context (sentiment + news).

Uses paraphrase-multilingual-MiniLM-L12-v2 for Spanish/English technical text.
Persistent storage in ./memory directory. Thread-safe for concurrent reads.
"""

import json
import os
import hashlib
from typing import Optional

from ..config import log

# ── Embedding model config ───────────────────────────────────
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "trading_strategies_v2"

# Sentinel file to track which embedding model was used
_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory")
_MODEL_SENTINEL = os.path.join(_MEMORY_DIR, ".embedding_model")

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False
    log.warning("[RAG] chromadb not installed — StrategyMemory disabled")

# ── SentenceTransformer embedding function for ChromaDB ──────
_embedding_fn = None

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=_EMBEDDING_MODEL_NAME,
    )
    log.info(f"[RAG] Embedding model loaded: {_EMBEDDING_MODEL_NAME}")
except ImportError:
    log.warning("[RAG] sentence-transformers not installed — using ChromaDB default embeddings")
except Exception as e:
    log.warning(f"[RAG] Failed to load {_EMBEDDING_MODEL_NAME}: {e} — using default")


def _check_model_migration(client) -> bool:
    """Detect if the embedding model changed. If so, delete old collection.

    Returns True if a migration happened (caller should re-ingest).
    """
    os.makedirs(_MEMORY_DIR, exist_ok=True)
    sentinel_path = _MODEL_SENTINEL

    current_model = _EMBEDDING_MODEL_NAME if _embedding_fn else "default"

    # Read previous model from sentinel
    previous_model = None
    if os.path.exists(sentinel_path):
        try:
            with open(sentinel_path, "r") as f:
                previous_model = f.read().strip()
        except Exception:
            pass

    if previous_model == current_model:
        return False  # Same model, no migration needed

    # Model changed — purge old collections
    log.warning(f"[RAG] Embedding model changed: {previous_model!r} → {current_model!r}")
    try:
        existing = [c.name for c in client.list_collections()]
        for cname in existing:
            if cname.startswith("trading_strategies"):
                client.delete_collection(cname)
                log.info(f"[RAG] Deleted old collection: {cname}")
    except Exception as e:
        log.error(f"[RAG] Failed to purge old collections: {e}")

    # Write new sentinel
    try:
        with open(sentinel_path, "w") as f:
            f.write(current_model)
    except Exception as e:
        log.warning(f"[RAG] Failed to write model sentinel: {e}")

    return True


class StrategyMemory:
    """Vector store for institutional trading strategies.

    Each strategy is a JSON object with at least:
      - strategy_name: str
      - optimal_context: str  (market conditions where this strategy excels)
      - entry_rules: str      (concrete entry criteria)

    Retrieval uses paraphrase-multilingual-MiniLM-L12-v2 for accurate
    Spanish/English matching of market context against strategies.
    """

    def __init__(self, persist_dir: str = _MEMORY_DIR):
        self._collection = None
        self._needs_reingest = False
        if not _CHROMADB_AVAILABLE:
            return

        try:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # Check if embedding model changed → purge + re-ingest
            self._needs_reingest = _check_model_migration(self._client)

            # Create collection with the multilingual embedding function
            kwargs = {
                "name": _COLLECTION_NAME,
                "metadata": {"hnsw:space": "cosine"},
            }
            if _embedding_fn is not None:
                kwargs["embedding_function"] = _embedding_fn

            self._collection = self._client.get_or_create_collection(**kwargs)
            count = self._collection.count()

            if self._needs_reingest or count == 0:
                log.info(f"[RAG] Collection empty or migrated — needs re-ingestion")
                self._needs_reingest = True
            else:
                log.info(f"[RAG] StrategyMemory loaded — {count} strategies "
                         f"(model: {_EMBEDDING_MODEL_NAME})")
        except Exception as e:
            log.error(f"[RAG] Failed to initialize ChromaDB: {e}")
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    @property
    def needs_reingest(self) -> bool:
        return self._needs_reingest

    def add_strategy(self, json_data: str) -> bool:
        """Add a distilled strategy to the vector store.

        Args:
            json_data: JSON string with strategy_name, optimal_context, entry_rules.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.available:
            log.warning("[RAG] add_strategy called but ChromaDB not available")
            return False

        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
        except json.JSONDecodeError as e:
            log.error(f"[RAG] Invalid JSON in add_strategy: {e}")
            return False

        name = data.get("strategy_name", "unknown")
        # Build rich searchable document from all strategy fields
        context = data.get("optimal_context", "")
        if isinstance(context, dict):
            context = ", ".join(f"{k}: {v}" for k, v in context.items())
        rules = data.get("entry_rules", "")
        if isinstance(rules, list):
            rules = " | ".join(rules)
        family = data.get("concept_family", "")
        invalidation = data.get("invalidation_condition", "")

        document = (
            f"Estrategia: {name}. "
            f"Familia: {family}. "
            f"Contexto óptimo: {context}. "
            f"Reglas de entrada: {rules}. "
            f"Invalidación: {invalidation}"
        )

        # Deterministic ID from strategy name (upsert semantics)
        doc_id = hashlib.md5(name.encode()).hexdigest()

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[{"json": json.dumps(data, ensure_ascii=False)}],
            )
            log.info(f"[RAG] Strategy stored: {name}")
            return True
        except Exception as e:
            log.error(f"[RAG] Failed to store strategy '{name}': {e}")
            return False

    def get_best_strategy(self, market_context: str,
                          n_results: int = 1,
                          exclude_strategies: list[str] | None = None,
                          ) -> Optional[dict]:
        """Retrieve the most relevant strategy for the current market context.

        Args:
            market_context: String describing current sentiment, news, F&G, etc.
            n_results: Number of top results to consider (returns best).
            exclude_strategies: List of strategy names to skip (fatigue blacklist).

        Returns:
            Strategy dict (with strategy_name, optimal_context, entry_rules)
            or None if DB is empty / unavailable.
        """
        if not self.available:
            return None

        exclude = set(exclude_strategies or [])

        try:
            count = self._collection.count()
            if count == 0:
                return None

            # Fetch extra results if we need to skip blacklisted strategies.
            # Hard cap at 20 to prevent unbounded ChromaDB queries when many
            # strategies are blacklisted simultaneously.
            fetch_n = min(count, 20, max(n_results, len(exclude) + 3))
            results = self._collection.query(
                query_texts=[market_context],
                n_results=fetch_n,
            )

            if not results or not results.get("metadatas") or not results["metadatas"][0]:
                return None

            # Iterate results, skip blacklisted strategies
            for i, meta in enumerate(results["metadatas"][0]):
                json_str = meta.get("json", "{}")
                strategy = json.loads(json_str)
                strat_name = strategy.get("strategy_name", "")

                if strat_name in exclude:
                    log.info(f"[RAG] Skipping blacklisted strategy: {strat_name}")
                    continue

                distance = (results.get("distances", [[]])[0][i]
                            if results.get("distances") and len(results["distances"][0]) > i
                            else None)
                if distance is not None:
                    log.info(f"[RAG] Best strategy: {strat_name} "
                             f"(cosine distance={distance:.4f})"
                             f"{f' (skipped {i} blacklisted)' if i > 0 else ''}")
                else:
                    log.info(f"[RAG] Best strategy: {strat_name}")
                return strategy

            # All results were blacklisted
            log.info(f"[RAG] All top {fetch_n} strategies are blacklisted — no strategy")
            return None

        except Exception as e:
            log.error(f"[RAG] Query failed: {e}")
            return None

    def list_strategies(self) -> list[dict]:
        """List all stored strategies. Utility for dashboard/debug."""
        if not self.available:
            return []
        try:
            all_data = self._collection.get()
            strategies = []
            for meta in (all_data.get("metadatas") or []):
                try:
                    strategies.append(json.loads(meta.get("json", "{}")))
                except json.JSONDecodeError:
                    pass
            return strategies
        except Exception as e:
            log.error(f"[RAG] list_strategies failed: {e}")
            return []

    def count(self) -> int:
        """Number of strategies in the store."""
        if not self.available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
