"""
Ingestión de estrategias institucionales en ChromaDB (RAG).

Uso:
    python -m btc5min.ai.ingestar_rag

Lee archivos JSON de btc5min/strategies/rag/*.json e ingesta cada estrategia
en ChromaDB con upsert (idempotente por nombre).
"""
import json
from btc5min.ai.rag_db import StrategyMemory
from btc5min.strategies import load_rag_strategies

db = StrategyMemory()


def run_ingestion(force: bool = False, store: "StrategyMemory | None" = None):
    """Ingest all strategies from strategies/rag/*.json.

    Args:
        force: Force re-ingestion even if collection is not empty.
        store: Optional StrategyMemory instance to use (avoids creating
               a second PersistentClient). Falls back to module-level ``db``.
    """
    from btc5min.ai.rag_db import _EMBEDDING_MODEL_NAME

    target = store or db
    strategies = load_rag_strategies()
    print(f"Modelo de embeddings: {_EMBEDDING_MODEL_NAME}")
    print(f"Estrategias encontradas en strategies/rag/: {len(strategies)}")

    if target.needs_reingest:
        print("[!] Migracion de modelo detectada -- re-ingesta forzada")
    elif force:
        print("[!] Re-ingesta forzada por parametro")

    existing = target.count()
    print(f"Estrategias existentes en ChromaDB: {existing}")

    ok = 0
    fail = 0
    for est in strategies:
        try:
            json_str = json.dumps(est, ensure_ascii=False)
            target.add_strategy(json_str)
            print(f"  + {est['strategy_name']}")
            ok += 1
        except Exception as e:
            print(f"  ! Error: {e}")
            fail += 1

    print(f"\nResultado: {ok} guardadas, {fail} errores")
    print(f"Total de estrategias activas en memoria: {target.count()}")


if __name__ == "__main__" or not hasattr(__builtins__, "__IPYTHON__"):
    run_ingestion()
