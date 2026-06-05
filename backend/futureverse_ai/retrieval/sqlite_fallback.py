"""
SQLite Full-Text Search Fallback for RAG
=========================================
Creates a separate SQLite FTS5 index of all chunks.
When Weaviate returns 0 or low-confidence results,
this fallback does keyword search directly on chunk text.

DEPLOY:
  Copy to: D:\FutureverseAI\FutureverseAI\backend\futureverse_ai\retrieval\sqlite_fallback.py

HOW IT WORKS:
  1. When a file is uploaded → chunks are indexed in SQLite FTS5
  2. During retrieval → if Weaviate returns < min_results, SQLite fallback runs
  3. SQLite uses FTS5 full-text search (much better than LIKE)
  4. Results merged and deduplicated with Weaviate results
"""

import sqlite3
import os
import re
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Path to fallback index (separate from webui.db)
FALLBACK_DB = os.environ.get(
    "SQLITE_FALLBACK_DB",
    r"D:\FutureverseAI\FutureverseAI\backend\data\chunk_index.db"
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(FALLBACK_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_fallback_db():
    """Create FTS5 index table if not exists."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            file_id     TEXT NOT NULL,
            filename    TEXT,
            subject     TEXT,
            chapter     TEXT,
            class_level TEXT,
            text        TEXT NOT NULL,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            id UNINDEXED,
            file_id UNINDEXED,
            filename,
            subject,
            chapter,
            text,
            content='chunks',
            content_rowid='rowid'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai
        AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, id, file_id, filename, subject, chapter, text)
            VALUES (new.rowid, new.id, new.file_id, new.filename, new.subject, new.chapter, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad
        AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, id, file_id, filename, subject, chapter, text)
            VALUES ('delete', old.rowid, old.id, old.file_id, old.filename, old.subject, old.chapter, old.text);
        END;
    """)
    conn.commit()
    conn.close()
    log.info("SQLite FTS5 fallback index initialized")


def index_chunks(file_id: str, filename: str, chunks: list[dict]):
    """
    Index chunks from an uploaded file into SQLite FTS5.
    Call this after Weaviate indexing completes.
    
    chunks format: [{"text": "...", "metadata": {"subject": ..., "chapter": ...}}]
    """
    init_fallback_db()
    conn = get_connection()

    # Delete existing chunks for this file
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    inserted = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "") or chunk.get("content", "")
        if not text or len(text.split()) < 10:
            continue

        meta = chunk.get("metadata", {})
        chunk_id = f"{file_id}_{i:05d}"

        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, file_id, filename, subject, chapter, class_level, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            file_id,
            filename,
            meta.get("subject", ""),
            meta.get("chapter", ""),
            meta.get("class", ""),
            text,
        ))
        inserted += 1

    conn.commit()
    conn.close()
    log.info(f"Indexed {inserted} chunks for file {filename}")
    return inserted


def extract_keywords(query: str) -> str:
    """
    Extract meaningful keywords from a query for FTS5 search.
    Removes stop words, keeps content words.
    """
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
        'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'out', 'off', 'over', 'under',
        'which', 'what', 'who', 'how', 'when', 'where', 'why',
        'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
        'not', 'but', 'if', 'then', 'than', 'so', 'yet', 'both',
        'one', 'two', 'also', 'just', 'as', 'each', 'while',
        'following', 'given', 'above', 'consider', 'statements',
        'correct', 'incorrect', 'true', 'false', 'answer', 'option',
        'discussed', 'considered', 'many', 'all', 'none', 'only',
    }

    # Extract words from question and options
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
    keywords = [w.lower() for w in words if w.lower() not in STOP_WORDS]

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)

    # FTS5 OR query — find chunks with ANY of these keywords
    return " OR ".join(unique[:8])  # max 8 keywords


def fallback_search(
    query: str,
    top_k: int = 5,
    file_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Search SQLite FTS5 index for relevant chunks.
    
    Returns list of dicts compatible with Weaviate result format:
    [{"text": "...", "metadata": {...}, "source": "sqlite_fts"}]
    """
    if not os.path.exists(FALLBACK_DB):
        return []

    try:
        conn = get_connection()
        keywords = extract_keywords(query)

        if not keywords:
            return []

        log.info(f"SQLite FTS fallback search: '{keywords}'")

        if file_ids:
            placeholders = ",".join("?" * len(file_ids))
            rows = conn.execute(f"""
                SELECT c.id, c.file_id, c.filename, c.subject, c.chapter,
                       c.class_level, c.text,
                       rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.id = c.id
                WHERE chunks_fts MATCH ?
                AND c.file_id IN ({placeholders})
                ORDER BY rank
                LIMIT ?
            """, [keywords] + file_ids + [top_k]).fetchall()
        else:
            rows = conn.execute("""
                SELECT c.id, c.file_id, c.filename, c.subject, c.chapter,
                       c.class_level, c.text,
                       rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.id = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, [keywords, top_k]).fetchall()

        conn.close()

        results = []
        for row in rows:
            results.append({
                "text": row["text"],
                "metadata": {
                    "subject": row["subject"],
                    "chapter": row["chapter"],
                    "class": row["class_level"],
                    "source": row["filename"],
                    "file_id": row["file_id"],
                    "retrieval": "sqlite_fts_fallback",
                },
            })

        log.info(f"SQLite FTS returned {len(results)} fallback results")
        return results

    except Exception as e:
        log.warning(f"SQLite FTS fallback failed: {e}")
        return []


def get_stats() -> dict:
    """Return stats about the fallback index."""
    if not os.path.exists(FALLBACK_DB):
        return {"status": "not initialized", "chunks": 0, "files": 0}
    conn = get_connection()
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    files = conn.execute("SELECT COUNT(DISTINCT file_id) FROM chunks").fetchone()[0]
    conn.close()
    return {"status": "ok", "chunks": chunks, "files": files}


# ── Integration point in utils.py ────────────────────────────
INTEGRATION_INSTRUCTIONS = """
WHERE TO ADD IN utils.py (around line 1237 after query_results is built):

    # === SQLITE FTS FALLBACK ===
    try:
        # Count how many real chunks were retrieved
        total_retrieved = sum(
            len(qr.get("documents", [[]])[0])
            for qr in query_results
            if "documents" in qr
        )
        
        if total_retrieved < 3:  # fallback when Weaviate finds < 3 chunks
            from futureverse_ai.retrieval.sqlite_fallback import fallback_search
            fts_results = fallback_search(query=queries[0] if queries else "", top_k=5)
            
            if fts_results:
                query_results.append({
                    "documents": [[r["text"] for r in fts_results]],
                    "metadatas": [[r["metadata"] for r in fts_results]],
                    "file": {
                        "type": "sqlite_fts",
                        "id": "sqlite_fallback",
                        "name": "SQLite FTS Fallback",
                        "collection_name": "sqlite_fts",
                    }
                })
                log.info(f"SQLite FTS fallback added {len(fts_results)} chunks")
    except Exception as e:
        log.warning(f"SQLite FTS fallback error: {e}")
"""


if __name__ == "__main__":
    # Test the fallback
    print("Initializing SQLite FTS index...")
    init_fallback_db()
    print("Stats:", get_stats())

    # Test keyword extraction
    tests = [
        "Which fact was not considered in sea floor spreading concept?",
        "Amarkantak Hills are at confluence of which mountain ranges?",
        "National parks and sanctuaries are established for conservation",
    ]
    print("\nKeyword extraction test:")
    for q in tests:
        print(f"  Q: {q[:60]}")
        print(f"  K: {extract_keywords(q)}")
