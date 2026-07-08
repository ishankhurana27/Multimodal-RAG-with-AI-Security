"""
AKU (Atomic Knowledge Unit) embedding and chunk-linking endpoints.

  POST /api/v1/aku/embed   — Embed Q-Matrix AKU nodes as searchable documents
                             in the AkuKnowledge Weaviate collection.
  POST /api/v1/aku/link    — Retroactively link each AKU's NCERT anchor sentence
                             to the book chunk that contains it (patches aku_ids).

Both endpoints are called by the Agent Orchestration pipeline's `emb_aku_index`
stage, which runs after Q-Matrix generation completes.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_core.documents import Document
from pydantic import BaseModel
from sqlalchemy.orm import Session

from futureverse_ai.internal.db import get_session
from futureverse_ai.utils.auth import get_verified_user

log = logging.getLogger(__name__)
router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class AKUEmbedRequest(BaseModel):
    """Embed a batch of AKU nodes for one NCERT chapter."""
    chapter_id: str
    aku_nodes: list[dict]
    collection_name: str = "AkuKnowledge"


class AKULinkRequest(BaseModel):
    """Link each AKU's NCERT anchor to the matching book chunk."""
    chapter_id: str
    aku_nodes: list[dict]
    book_collection_name: str   # e.g. "file-{futureverse_file_id}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_ncert_anchor(aku: dict) -> str:
    """Extract the verbatim NCERT text used for BM25 chunk-linking.

    Priority:
      1. source_citation.verbatim_excerpt  — literal copy-paste from the book;
         highest BM25 recall because it appears word-for-word in the chunk.
      2. so_what_protocol.gate_5_measurability.evidence  — fallback when
         source_citation is absent (older Q-Matrix format).
    """
    # Priority 1: verbatim excerpt (literal book text)
    citation = aku.get("source_citation") or {}
    verbatim = (citation.get("verbatim_excerpt") or "").strip()
    if verbatim:
        return verbatim

    # Priority 2: gate_5_measurability evidence
    proto = aku.get("so_what_protocol") or {}
    gate5 = proto.get("gate_5_measurability") or {}
    return (gate5.get("evidence") or "").strip()


def _build_aku_rich_text(aku: dict) -> str:
    """Build the rich-text document that represents one AKU node for embedding.

    Combines the label, description, verbatim NCERT excerpt, wrong belief, and
    FITB sentence so that:
      • Semantic search on the verbatim excerpt matches the exact book passage.
      • Semantic search on the wrong belief matches PYQ distractor options.
      • Semantic search on the description matches paraphrased PYQ stems.
    """
    lines: list[str] = [
        f"# {aku.get('label', '')}",
        "",
        aku.get("description", ""),
        "",
    ]

    # Verbatim book text (source_citation preferred, gate_5 fallback)
    ncert = _get_ncert_anchor(aku)
    if ncert:
        lines += [f"**NCERT Verbatim:** {ncert}", ""]

    # Location note gives semantic context (e.g. "Para under 'Gram Sabha' section")
    citation = aku.get("source_citation") or {}
    location = (citation.get("location_note") or "").strip()
    if location:
        lines += [f"**Location:** {location}", ""]

    lines += [
        f"**Knowledge Structure:** {aku.get('knowledge_structure_type', '')}",
        f"**SOLO Level:**          {aku.get('solo_indicator', '')}",
        f"**Structural Role:**     {aku.get('structural_role', '')}",
    ]

    wrong_belief = aku.get("wrong_belief_specification")
    if wrong_belief:
        lines += ["", f"**Common Misconception:** {wrong_belief}"]

    fitb = aku.get("fitb_sentence")
    if fitb:
        lines += ["", f"**Diagnostic Sentence:** {fitb}"]

    return "\n".join(lines)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/embed")
def embed_aku_nodes(
    request: Request,
    form_data: AKUEmbedRequest,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    """
    Embed Q-Matrix AKU nodes as individual searchable documents.

    Each AKU becomes one document (``split=False`` — never split further).
    AKU-specific fields (aku_id, cluster_id, source_chapter_id, ncert_anchor,
    wrong_belief, etc.) are stored as non-vectorised Weaviate properties so
    they can be used as exact-match filters during PYQ mapping.

    Deduplication: re-uploading the same aku_id is a no-op (file_id = aku_{id}
    is already in the collection → ChunkDeduplication skips it).
    """
    from futureverse_ai.routers.retrieval import save_docs_to_vector_db

    if not form_data.aku_nodes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="aku_nodes list is empty",
        )

    docs: list[Document] = []
    for aku in form_data.aku_nodes:
        aku_id = aku.get("aku_id", "")
        if not aku_id:
            continue

        rich_text = _build_aku_rich_text(aku)
        meta: dict = {
            # AKU-specific (stored in Weaviate, not vectorised)
            "aku_id":            aku_id,
            "cluster_id":        aku.get("topic_cluster_id", ""),
            "source_chapter_id": form_data.chapter_id,
            "structural_role":   aku.get("structural_role", ""),
            "integration_node":  str(aku.get("integration_node", False)).lower(),
            "ncert_anchor":      _get_ncert_anchor(aku),
            "wrong_belief":      aku.get("wrong_belief_specification") or "",
            "fitb_sentence":     aku.get("fitb_sentence") or "",
            # Standard fields used by FutureverseAI dedup + collection scoping
            "file_id":           f"aku_{aku_id}",
            "source":            f"qmatrix_{form_data.chapter_id}",
            "name":              aku_id,
            "created_by":        str(getattr(user, "id", "system")),
        }
        docs.append(Document(page_content=rich_text, metadata=meta))

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid AKU nodes (all missing aku_id field)",
        )

    try:
        ok = save_docs_to_vector_db(
            request=request,
            docs=docs,
            collection_name=form_data.collection_name,
            split=False,   # AKU docs are atomic — never split
            add=True,
            user=user,
        )
    except Exception as exc:
        log.exception("AKU embed failed for chapter=%s: %s", form_data.chapter_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )

    return {
        "status":     "ok" if ok else "error",
        "chapter_id": form_data.chapter_id,
        "collection": form_data.collection_name,
        "embedded":   len(docs),
    }


@router.post("/link")
def link_aku_to_chunks(
    request: Request,
    form_data: AKULinkRequest,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    """
    Link each AKU to the book chunk that contains its NCERT anchor sentence.

    For each AKU node:
      1. BM25-search the book chunk collection using the verbatim NCERT anchor.
      2. Take the top hit (highest BM25 score).
      3. PATCH that chunk's ``aku_ids`` field (comma-separated list) — the
         vector is NOT re-generated; only the scalar property is updated.

    A single book chunk may cover multiple AKU nodes (e.g. one paragraph that
    contains two distinct propositions); this is handled by appending to the
    existing ``aku_ids`` string rather than overwriting it.

    AKUs with short or missing NCERT anchors are skipped silently.
    """
    from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT

    if not form_data.aku_nodes:
        return {"status": "ok", "links_created": 0, "skipped": 0}

    links_created  = 0
    skipped        = 0
    chunk_mappings: list[dict] = []   # [{aku_id, chunk_id, chunk_text, bm25_score}]

    for aku in form_data.aku_nodes:
        aku_id       = aku.get("aku_id", "")
        ncert_anchor = _get_ncert_anchor(aku)

        if not aku_id or not ncert_anchor or len(ncert_anchor) < 15:
            skipped += 1
            continue

        # BM25 search: find the chunk most likely containing this NCERT sentence
        result = VECTOR_DB_CLIENT.bm25_search(
            collection_name=form_data.book_collection_name,
            query=ncert_anchor,
            query_properties=["raw_text", "text"],
            limit=3,
        )
        if not result or not result.ids or not result.ids[0]:
            skipped += 1
            log.debug("aku_link: no chunk found for aku_id=%s anchor=%s…", aku_id, ncert_anchor[:60])
            continue

        top_uuid      = result.ids[0][0]
        existing_meta = (result.metadatas[0][0] or {}) if result.metadatas else {}
        existing_ids  = existing_meta.get("aku_ids", "")
        top_score     = (result.distances[0][0] if result.distances else 0.0) or 0.0
        top_text      = (result.documents[0][0] if result.documents else "") or ""

        # Append only if not already present
        current_set = set(existing_ids.split(",")) if existing_ids else set()
        if aku_id not in current_set:
            current_set.add(aku_id)
            new_aku_ids = ",".join(sorted(current_set))
            patched = VECTOR_DB_CLIENT.update_object_properties(
                collection_name=form_data.book_collection_name,
                object_id=top_uuid,
                properties={"aku_ids": new_aku_ids},
            )
            if patched:
                links_created += 1
                log.debug("aku_link: linked aku_id=%s → chunk %s", aku_id, top_uuid)
            else:
                skipped += 1
                continue
        else:
            # Already linked — still record the mapping so caller can write it to MongoDB
            log.debug("aku_link: aku_id=%s already in chunk %s", aku_id, top_uuid)

        # Record chunk mapping regardless of whether the patch was new or pre-existing
        chunk_mappings.append({
            "aku_id":      aku_id,
            "chunk_id":    top_uuid,
            "chunk_text":  top_text,
            "bm25_score":  round(float(top_score), 4),
        })

    return {
        "status":         "ok",
        "chapter_id":     form_data.chapter_id,
        "collection":     form_data.book_collection_name,
        "links_created":  links_created,
        "skipped":        skipped,
        "chunk_mappings": chunk_mappings,   # written back to Q-Matrix by caller
    }
