import logging
import os
from typing import Awaitable, Optional, Union

import requests
import aiohttp
import asyncio
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
import time
import re

from urllib.parse import quote
from huggingface_hub import snapshot_download
from langchain_classic.retrievers import (
    EnsembleRetriever,
)

from langchain_core.documents import Document

from futureverse_ai.config import VECTOR_DB
from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT


from futureverse_ai.models.users import UserModel
from futureverse_ai.models.files import Files
from futureverse_ai.models.knowledge import Knowledges

from futureverse_ai.models.chats import Chats
from futureverse_ai.models.notes import Notes
from futureverse_ai.models.access_grants import AccessGrants

from futureverse_ai.retrieval.vector.main import GetResult
from futureverse_ai.utils.headers import include_user_info_headers
from futureverse_ai.utils.misc import get_message_list

from futureverse_ai.retrieval.web.utils import get_web_loader
from futureverse_ai.retrieval.loaders.youtube import YoutubeLoader
from sentence_transformers import SentenceTransformer
import asyncio
import torch


from futureverse_ai.env import (
    AIOHTTP_CLIENT_TIMEOUT,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    AIOHTTP_CLIENT_SESSION_SSL,
)
from futureverse_ai.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
)

log = logging.getLogger(__name__)


from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


def is_youtube_url(url: str) -> bool:
    youtube_regex = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$"
    return re.match(youtube_regex, url) is not None


def get_loader(request, url: str):
    if is_youtube_url(url):
        return YoutubeLoader(
            url,
            language=request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            proxy_url=request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
        )
    else:
        return get_web_loader(
            url,
            verify_ssl=request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            requests_per_second=request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            trust_env=request.app.state.config.WEB_SEARCH_TRUST_ENV,
        )


def get_content_from_url(request, url: str) -> str:
    loader = get_loader(request, url)
    docs = loader.load()
    content = " ".join([doc.page_content for doc in docs])
    return content, docs


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int
    metadata_filter: Optional[dict] = None
    pre_computed_embedding: Optional[list] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self.pre_computed_embedding is not None:
            embedding = self.pre_computed_embedding
        else:
            embedding = await self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[embedding],
            limit=self.top_k,
            filter=self.metadata_filter or None,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            # Use raw_text (original chunk without [Context: ...] prefix) for LLM display.
            # The contextual prefix lives in `documents[idx]` (embedded field) and is
            # intentionally excluded from what the model sees — it's only for retrieval.
            display_text = metadatas[idx].get("raw_text") or documents[idx]
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=display_text,
                )
            )
        return results


class WeaviateBM25Retriever(BaseRetriever):
    """Replaces LangChain's in-memory BM25Retriever with Weaviate's native
    inverted-index BM25. Pre-indexed at write time → millisecond lookups even
    on 100k+ chunks. Searches raw_text + keywords fields (never the embedded
    [Context: ...] prefix stored in the `text` field)."""

    collection_name: Any
    top_k: int = 15
    query_properties: list = ["raw_text", "keywords"]
    metadata_filter: Optional[dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = VECTOR_DB_CLIENT.bm25_search(
            collection_name=self.collection_name,
            query=query,
            query_properties=self.query_properties,
            limit=self.top_k,
            filter=self.metadata_filter or None,
        )
        if not result or not result.documents or not result.documents[0]:
            return []
        docs = []
        for i, text in enumerate(result.documents[0]):
            meta = result.metadatas[0][i] if result.metadatas and result.metadatas[0] else {}
            score = result.distances[0][i] if result.distances and result.distances[0] else 0.0
            docs.append(Document(page_content=text, metadata={**meta, "bm25_score": score}))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def query_doc(
    collection_name: str, query_embedding: list[float], k: int, user: UserModel = None
):
    try:
        import time as _t
        _t0     = _t.monotonic()
        result  = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )
        _ms     = (_t.monotonic() - _t0) * 1000
        n_hits  = len(result.ids[0]) if result and result.ids else 0
        print(
            f"  VECTOR  {collection_name[:40]:<40}  hits={n_hits:>3}  ({_ms:.0f}ms)",
            flush=True,
        )
        return result
    except Exception as e:
        print(f"  VECTOR  {collection_name[:40]}  ERROR: {e}", flush=True)
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(collection_name: str, user: UserModel = None):
    try:
        log.debug(f"get_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def _build_context_header(meta: dict) -> str:
    """
    Build the Strategy B structured header for LLM consumption.

    Reads fields stored in metadata during ingestion (hierarchical+contextual pipeline).
    Called only when enable_enriched_texts=True AND has_context=True.
    Returns "" if the required fields are absent (graceful fallback for old chunks).

    Exact output format:
        [Source: {Header 1} > {Header 2}, paragraph {pos} of {total}]
        [Topic: {topics}]

    Rules:
    - position_in_section is 0-indexed in metadata → display as 1-indexed
    - Header 2 omitted when absent; Header 1 falls back to document filename
    - [Topic: ...] line omitted entirely when topics is empty/None
    - total derived from child_chunk_ids comma-count (all children under parent)
    """
    h1         = (meta.get("Header 1") or meta.get("header_1") or "").strip()
    h2         = (meta.get("Header 2") or meta.get("header_2") or "").strip()
    breadcrumb = (meta.get("breadcrumb") or "").strip()
    filename   = (meta.get("name") or meta.get("title") or meta.get("source") or "").strip()

    if h1 and h2:
        source_label = f"{h1} > {h2}"
    elif breadcrumb:
        source_label = breadcrumb          # already "H1 > H2 > H3" formatted
    elif h1:
        source_label = h1
    elif filename:
        source_label = filename
    else:
        return ""  # not enough metadata to build a meaningful header

    pos         = meta.get("position_in_section") or 0
    display_pos = int(pos) + 1

    child_ids_str = (meta.get("child_chunk_ids") or "").strip()
    if child_ids_str:
        total = len([c for c in child_ids_str.split(",") if c.strip()])
        pos_str = f"paragraph {display_pos} of {total}"
    else:
        pos_str = f"paragraph {display_pos}"

    source_line = f"[Source: {source_label}, {pos_str}]"

    topics     = (meta.get("topics") or "").strip()
    topic_line = f"[Topic: {topics}]" if topics else ""

    lines = [source_line]
    if topic_line:
        lines.append(topic_line)
    return "\n".join(lines)


def get_enriched_texts(collection_result: GetResult) -> list[str]:
    enriched_texts = []
    for idx, text in enumerate(collection_result.documents[0]):
        metadata = collection_result.metadatas[0][idx]
        # Use raw_text (original chunk without context prefix) for display and BM25.
        # The contextual text lives in `text` (the embedded field) — we never show
        # the "[Context: ...]" prefix to users; search still benefits from it via
        # the vector index.
        display_text = metadata.get("raw_text") or text
        metadata_parts = [display_text]

        # Add filename (repeat twice for extra weight in BM25 scoring)
        if metadata.get("name"):
            filename = metadata["name"]
            filename_tokens = (
                filename.replace("_", " ").replace("-", " ").replace(".", " ")
            )
            metadata_parts.append(
                f"Filename: {filename} {filename_tokens} {filename_tokens}"
            )

        # Add title if available
        if metadata.get("title"):
            metadata_parts.append(f"Title: {metadata['title']}")

        # Add document section headings if available (from markdown splitter)
        if metadata.get("headings") and isinstance(metadata["headings"], list):
            headings = " > ".join(str(h) for h in metadata["headings"])
            metadata_parts.append(f"Section: {headings}")

        # Add source URL/path if available
        if metadata.get("source"):
            metadata_parts.append(f"Source: {metadata['source']}")

        # For hierarchical chunks: include parent text in BM25 corpus so that
        # section-level terms can match even when querying a child chunk.
        if metadata.get("parent_text"):
            metadata_parts.append(metadata["parent_text"])

        # Add snippet for web search results
        if metadata.get("snippet"):
            metadata_parts.append(f"Snippet: {metadata['snippet']}")
        if metadata.get("chapter"):
            metadata_parts.append(f"Chapter: {metadata['chapter']}")

        if metadata.get("class_level"):
            metadata_parts.append(f"NCERT Class: {metadata['class_level']}")

        if metadata.get("subject"):
            metadata_parts.append(f"Subject: {metadata['subject']}")

        enriched_texts.append(" ".join(metadata_parts))

    return enriched_texts


def _build_metadata_filter(query: str, sample_metadatas: list) -> dict:
    """Detect subject / class_level signals in the query and build a
    pre-filter dict for Weaviate.

    Safety rule: a field is only included in the filter when the collection
    sample contains at least one document with that field populated — a proxy
    for "this collection actually uses this field".  This prevents accidentally
    filtering on a field that was never ingested, which would return zero
    results with no warning.

    Returns a plain {field: value} dict (empty = no filter).
    Multiple entries are ANDed by _build_weaviate_filter in weaviate.py.
    """
    _has_subject = any(m.get("subject") for m in sample_metadatas if isinstance(m, dict))
    _has_class   = any(m.get("class_level") for m in sample_metadatas if isinstance(m, dict))

    if not _has_subject and not _has_class:
        return {}

    filter_dict: dict = {}
    q_lower = query.lower()

    if _has_subject:
        subject_map = {
            "chemistry":          "Chemistry",
            "physics":            "Physics",
            "biology":            "Biology",
            "history":            "History",
            "geography":          "Geography",
            "economics":          "Economics",
            "political science":  "Political Science",
            "civics":             "Civics",
            "mathematics":        "Mathematics",
            "maths":              "Mathematics",
            "science":            "Science",
            "english":            "English",
            "hindi":              "Hindi",
        }
        for kw, subject in subject_map.items():
            if kw in q_lower:
                filter_dict["subject"] = subject
                break

    if _has_class:
        _m = re.search(
            r'\b(?:class|grade|std)\.?\s*(10|11|12|9|8)\b'
            r'|\b(10|11|12|9|8)(?:th|st|rd|nd)\s+(?:class|grade|std)\b',
            q_lower,
        )
        if _m:
            filter_dict["class_level"] = _m.group(1) or _m.group(2)

    return filter_dict


async def query_doc_with_hybrid_search(
    collection_name: str,
    collection_sample: Optional[GetResult],
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    enable_enriched_texts: bool = False,
    pre_computed_embedding: Optional[list] = None,
) -> dict:
    """Hybrid BM25 + vector search.

    collection_sample: a small sample (≤10 items) used only for null-checking and
    detecting hierarchical collections. BM25 now runs natively inside Weaviate —
    no full collection scan needed.
    """
    try:
        # Null check on the sample (collection empty or missing)
        if (
            not collection_sample
            or not hasattr(collection_sample, "documents")
            or not collection_sample.documents
            or not collection_sample.documents[0]
        ):
            log.warning(f"query_doc_with_hybrid_search:no_docs {collection_name}")
            return {"documents": [], "metadatas": [], "distances": []}

        log.debug(f"query_doc_with_hybrid_search:doc {collection_name}")

        # Metadata pre-filter: narrow the Weaviate search to the relevant subject /
        # class_level before BM25 and vector scoring begins.  Verified against the
        # collection sample so we never apply a filter on a field that was never
        # ingested (which would silently return 0 results).
        _sample_metas = (
            collection_sample.metadatas[0]
            if collection_sample and collection_sample.metadatas
            else []
        )
        _meta_filter = _build_metadata_filter(query, _sample_metas)
        if _meta_filter:
            log.debug(
                "[MetaFilter] pre-filter applied: %s  collection=%s",
                _meta_filter, collection_name,
            )

        # Native Weaviate BM25 — searches the pre-built inverted index directly.
        # Searches raw_text + keywords fields (not the [Context: ...] embedded text).
        bm25_retriever = WeaviateBM25Retriever(
            collection_name=collection_name,
            top_k=min(k, 15),
            query_properties=["raw_text", "keywords"],
            metadata_filter=_meta_filter or None,
        )

        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
            metadata_filter=_meta_filter or None,
            pre_computed_embedding=pre_computed_embedding,
        )

        # Dynamic weighting — boost BM25 for short queries OR specific
        # entity names (capitalised terms).
        words = query.split()
        has_entity = any(w[0].isupper() for w in words if len(w) > 3)
        if len(words) <= 4 or has_entity:
            hybrid_bm25_weight = 0.65  # BM25 leads for entity/keyword queries
        else:
            hybrid_bm25_weight = 0.35  # Vector leads for conceptual queries

        # Hierarchical collections contain small child chunks (~CHUNK_SIZE/4).
        # Enforce BM25 ≥ 0.65 for higher keyword precision on small chunks.
        is_hierarchical = any(
            m.get("parent_id")
            for m in (collection_sample.metadatas[0] if collection_sample.metadatas else [])
        )
        if is_hierarchical:
            hybrid_bm25_weight = max(hybrid_bm25_weight, 0.65)

        if hybrid_bm25_weight <= 0:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_search_retriever], weights=[1.0]
            )
        elif hybrid_bm25_weight >= 1:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever], weights=[1.0]
            )
        else:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_search_retriever],
                weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
            )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=max(k_reranker, 15),
            reranking_function=reranking_function,
            r_score=r,
        )

        # Issue 5: single dedup pass before cross-encoder — correct abstraction.
        # Step 1: EnsembleRetriever returns raw candidates (BM25 + vector arms may
        #         return the same chunk). ContextualCompressionRetriever removed —
        #         its only role was wiring ensemble→compressor; splitting the steps
        #         gives us the interception point needed for pre-rerank dedup.
        # Step 2: Dedup on raw_text hash — single pass, keeps first occurrence
        #         (highest RRF score). No second pass after reranking: reranker
        #         already saw unique documents and its scores are the final authority.
        # Step 3: Pass unique candidates to reranker.
        _t_retrieval = time.perf_counter()
        raw_candidates = await ensemble_retriever.ainvoke(query)

        _seen_ids: set = set()
        _deduped_candidates: list = []
        for _dc in raw_candidates:
            _raw_dc = _dc.metadata.get("raw_text") or _dc.page_content
            _h_dc = hashlib.sha256(_raw_dc.encode()).hexdigest()
            if _h_dc not in _seen_ids:
                _seen_ids.add(_h_dc)
                _deduped_candidates.append(_dc)
        _pre_dedup_dropped = len(raw_candidates) - len(_deduped_candidates)
        if _pre_dedup_dropped:
            log.debug(
                "Pre-rerank dedup: dropped %d/%d duplicates before cross-encoder",
                _pre_dedup_dropped, len(raw_candidates),
            )

        log.debug(
            "[Reranker] Stage1 candidates=%d collection=%s",
            len(_deduped_candidates), collection_name,
        )
        _t_rerank = time.perf_counter()
        _s1_t0 = _t_rerank
        result = await compressor.acompress_documents(_deduped_candidates, query)
        log.debug(
            "[Reranker] Stage1 latency=%.0fms candidates=%d collection=%s",
            (time.perf_counter() - _s1_t0) * 1000,
            len(_deduped_candidates), collection_name,
        )

        # Parent dedup: multiple children from the same parent all expand to the same
        # parent_text body. Keep only the highest-scoring child per parent_id so the
        # LLM never sees the same section text repeated. Comparison key is the reranker
        # score stored in metadata["score"] (sigmoid-normalized + metadata bonuses).
        _parent_best: dict = {}
        for _d in result:
            _pid = _d.metadata.get("parent_id")
            if not _pid:
                continue
            _sc = _d.metadata.get("score", 0.0)
            if _pid not in _parent_best or _sc > _parent_best[_pid].metadata.get("score", 0.0):
                _parent_best[_pid] = _d

        # Second pass: walk result in reranker rank order (best first).
        # The parent surfaces at the position of its first-encountered child
        # (i.e. highest retrieval rank), but is represented by the child that
        # scored highest in _parent_best. This is intentional: position reflects
        # retrieval salience; metadata/score reflects semantic quality. A "fix"
        # that substitutes the winner's position would break rank ordering.
        _result_deduped: list = []
        _seen_pids: set = set()
        _parent_dedup_collapsed = 0
        for _d in result:
            _pid = _d.metadata.get("parent_id")
            if _pid:
                if _pid not in _seen_pids:
                    _seen_pids.add(_pid)
                    _result_deduped.append(_parent_best[_pid])
                else:
                    _parent_dedup_collapsed += 1
            else:
                _result_deduped.append(_d)

        if _parent_dedup_collapsed:
            log.debug(
                "Parent dedup: collapsed %d sibling chunk(s) into %d unique parent(s)"
                " — kept highest reranker score per parent_id",
                _parent_dedup_collapsed, len(_seen_pids),
            )
        result = _result_deduped

        if not result:
            log.warning(
                "[Retrieval] No chunks survived threshold filter "
                "(collection=%s, threshold=%.2f, query=%r)",
                collection_name,
                r if r is not None else 0.3,
                query[:200],
            )
            return {
                "documents": [["__NO_RESULTS__"]],
                "metadatas": [[{"retrieval_failed": True}]],
                "distances": [[0.0]],
            }

        distances = [(d.metadata.get("score") or 0.0) for d in result]  # guard None (Phase 1, M-10)

        # Build LLM-facing document content.
        # Base content priority: parent_text (hierarchical) → raw_text (contextual) → page_content.
        # parent_text gives the LLM the full section context (4× child size).
        # raw_text strips the [CONTEXT]/[CONTENT] embedding artifacts from page_content.
        #
        # Issue 1 — Strategy B context header (enable_enriched_texts=True):
        # When the chunk was contextually enriched (has_context=True), prepend a clean
        # structured header so the LLM knows WHERE in the document the chunk comes from
        # and WHAT topic it covers — without exposing the raw [CONTEXT] embedding prefix.
        # Chunks ingested before contextual chunking (has_context missing/False) fall back
        # to bare parent_text — no crash, no header, no degradation.
        _header_skipped = 0
        documents = []
        for d in result:
            body = (
                d.metadata.get("parent_text")
                or d.metadata.get("raw_text")
                or d.page_content
            )
            # Strip any [CONTEXT]...[CONTENT] block stored inside parent_text
            # at ingestion time — prevents the LLM seeing the raw embedding
            # prefix on top of the Strategy B structured header.
            body = re.sub(r'\[CONTEXT\].*?\[CONTENT\]\n?', '', body, flags=re.DOTALL).strip()
            if enable_enriched_texts:
                if d.metadata.get("has_context"):
                    header = _build_context_header(d.metadata)
                    if header:
                        body = f"{header}\n{body}"
                        log.debug(
                            "[ContextHeader] chunk pos=%s header_bytes=%d",
                            d.metadata.get("position_in_section", "?"),
                            len(header),
                        )
                    else:
                        _header_skipped += 1
                else:
                    _header_skipped += 1
            documents.append(body)

        if enable_enriched_texts and _header_skipped:
            log.debug(
                "[ContextHeader] context_header_skipped_count=%d "
                "(has_context=False or missing source metadata)",
                _header_skipped,
            )

        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]

            if sorted_items:
                distances, documents, metadatas = map(list, zip(*sorted_items))
            else:
                distances, documents, metadatas = [], [], []

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }
        log.info("Top retrieved chunks:")

        for d in documents[:5]:
            log.info(d[:200])

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e


def merge_get_results(get_results: list[dict]) -> dict:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def merge_and_sort_query_results(query_results: list[dict], k: int) -> dict:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes

    for data in query_results:
        if (
            len(data.get("distances", [])) == 0
            or len(data.get("documents", [])) == 0
            or len(data.get("metadatas", [])) == 0
        ):
            continue

        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (distance, document, metadata)
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if distance > combined[doc_hash][0]:
                    combined[doc_hash] = (distance, document, metadata)

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)

    # Slice to keep only the top k elements
    sorted_distances, sorted_documents, sorted_metadatas = (
        zip(*combined[:k]) if combined else ([], [], [])
    )

    # Create and return the output dictionary
    return {
        "distances": [list(sorted_distances)],
        "documents": [list(sorted_documents)],
        "metadatas": [list(sorted_metadatas)],
    }


def get_all_items_from_collections(collection_names: list[str]) -> dict:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


async def query_collection(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
) -> dict:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = await embedding_function(
        queries, prefix=RAG_EMBEDDING_QUERY_PREFIX
    )
    log.debug(
        f"query_collection: processing {len(queries)} queries across {len(collection_names)} collections"
    )

    with ThreadPoolExecutor() as executor:
        future_results = []
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                result = executor.submit(
                    process_query_collection, collection_name, query_embedding
                )
                future_results.append(result)
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


async def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    enable_enriched_texts: bool = False,
) -> dict:
    results = []
    error = False
    # Fetch collection data once per collection sequentially
    # Avoid fetching the same data multiple times later
    # peek() fetches ≤10 items — O(1) vs O(n) for get().
    # Used only for null-checks and hierarchical detection; BM25 is now native in Weaviate.
    collection_results = {}
    for collection_name in collection_names:
        try:
            log.debug(
                f"query_collection_with_hybrid_search:peek:collection {collection_name}"
            )
            collection_results[collection_name] = VECTOR_DB_CLIENT.peek(
                collection_name=collection_name, limit=10
            )
        except Exception as e:
            log.exception(f"Failed to peek collection {collection_name}: {e}")
            collection_results[collection_name] = None

    log.info(
        f"Starting hybrid search for {len(queries)} queries in {len(collection_names)} collections..."
    )

    # Semaphore: cap concurrent GPU embedding calls to avoid OOM / saturation.
    # With 7 collections × N queries in gather(), without a cap all tasks fire
    # their embedding requests simultaneously and crash the GPU.
    _sem = asyncio.Semaphore(4)

    # Eager batch pre-compute: embed all unique queries in one bge-m3 batch call
    # before launching tasks. Replaces N sequential single-string embedding calls
    # (one per VectorSearchRetriever invocation) with a single batched forward pass.
    _unique_queries = list(dict.fromkeys(queries))  # preserve order, deduplicate
    # Cross-request query-embedding cache (Phase 3) — default OFF, fail-open, model-keyed.
    from futureverse_ai.retrieval.embedding_cache import (
        CACHE_ENABLED as _EMB_CACHE_ON,
        get_cached as _emb_get,
        set_cached as _emb_set,
    )
    _cached = _emb_get(_unique_queries) if _EMB_CACHE_ON else {}
    _missing = [q for q in _unique_queries if q not in _cached]
    if _missing:
        _fresh_map = dict(
            zip(_missing, await embedding_function(_missing, RAG_EMBEDDING_QUERY_PREFIX))
        )
        if _EMB_CACHE_ON:
            _emb_set(_fresh_map)
    else:
        _fresh_map = {}
    _embedding_cache = {**_cached, **_fresh_map}

    async def process_query(collection_name, query):
        async with _sem:
            try:
                result = await query_doc_with_hybrid_search(
                    collection_name=collection_name,
                    collection_sample=collection_results[collection_name],
                    query=query,
                    embedding_function=embedding_function,
                    k=k,
                    reranking_function=reranking_function,
                    k_reranker=k_reranker,
                    r=r,
                    hybrid_bm25_weight=hybrid_bm25_weight,
                    enable_enriched_texts=enable_enriched_texts,
                    pre_computed_embedding=_embedding_cache.get(query),
                )
                return result, None
            except Exception as e:
                log.exception(f"Error when querying the collection with hybrid_search: {e}")
                return None, e

    # Prepare tasks for collections that have BM25 data (collection_result is not None).
    # Collections where get() returned None (empty or missing) are handled below via
    # pure vector search fallback, so they are NOT silently dropped.
    hybrid_tasks = [
        (collection_name, query)
        for collection_name in collection_names
        if collection_results[collection_name] is not None
        for query in queries
    ]
    vector_only_collections = [
        col for col in collection_names if collection_results[col] is None
    ]

    if vector_only_collections:
        log.info(
            "Collections with no BM25 data (falling back to vector-only): %s",
            vector_only_collections,
        )

    # Run hybrid tasks — semaphore inside process_query limits GPU concurrency to 4
    task_results = await asyncio.gather(
        *[process_query(collection_name, query) for collection_name, query in hybrid_tasks]
    )

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    # Vector-only fallback for collections that had no BM25 data
    if vector_only_collections:
        try:
            vector_fallback = await query_collection(
                collection_names=vector_only_collections,
                queries=queries,
                embedding_function=embedding_function,
                k=k,
            )
            if vector_fallback and vector_fallback.get("documents", [[]])[0]:
                results.append(vector_fallback)
                log.info(
                    "Vector-only fallback retrieved %d chunks from %s",
                    len(vector_fallback["documents"][0]),
                    vector_only_collections,
                )
        except Exception as _vf_err:
            log.warning("Vector-only fallback failed: %s", _vf_err)

    if error and not results:
        raise Exception(
            "Hybrid search failed for all collections. Using Non-hybrid search as fallback."
        )

    return merge_and_sort_query_results(results, k=k)


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        r = requests.post(
            f"{url}/embeddings",
            headers=headers,
            json=json_data,
            timeout=(10, 120),  # (connect, read) — never hang the worker thread
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            # Was `raise "<str>"` (TypeError in Py3, masked the real failure).
            raise RuntimeError(f"OpenAI embeddings response missing 'data': {data}")
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


async def agenerate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"agenerate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        form_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            form_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(
                f"{url}/embeddings",
                headers=headers,
                json=form_data,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                r.raise_for_status()
                data = await r.json()
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            headers = {
                "Content-Type": "application/json",
                "api-key": key,
            }
            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)

            r = requests.post(
                url,
                headers=headers,
                json=json_data,
                timeout=(10, 120),  # (connect, read) — never hang the worker thread
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


async def agenerate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"agenerate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        form_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            form_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        full_url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": key,
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(
                full_url,
                headers=headers,
                json=form_data,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                r.raise_for_status()
                data = await r.json()
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        r = requests.post(
            f"{url}/api/embed",
            headers=headers,
            json=json_data,
            timeout=(10, 120),  # (connect, read) — never hang the worker thread
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            # Was `raise "<str>"` (TypeError in Py3, masked the real failure).
            raise RuntimeError(f"Ollama embeddings response missing 'embeddings': {data}")
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


async def agenerate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"agenerate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        form_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            form_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(
                f"{url}/api/embed",
                headers=headers,
                json=form_data,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                r.raise_for_status()
                data = await r.json()
                if "embeddings" in data:
                    return data["embeddings"]
                else:
                    raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
    enable_async=True,
) -> Awaitable:
    # "" = local sentence-transformers; "sentence_transformers" is the DB-stored alias for the same engine
    if embedding_engine in ("", "sentence_transformers"):
        if embedding_function is None:
            log.error(
                f"get_embedding_function: embedding_function is None for engine='{embedding_engine}' "
                f"model='{embedding_model}'. SentenceTransformer did not load — "
                "check startup logs for the root cause."
            )

            async def failed_embedding_function(query, prefix=None, user=None):
                raise RuntimeError(
                    f"Embedding model '{embedding_model}' is not loaded "
                    f"(engine='{embedding_engine}'). "
                    "Check backend startup logs for SentenceTransformer initialization errors."
                )

            return failed_embedding_function

        # Sentence transformers: CPU-bound sync operation
        async def async_embedding_function(query, prefix=None, user=None):
            return await asyncio.to_thread(
                (
                    lambda query, prefix=None: embedding_function.encode(
                        query,
                        batch_size=int(embedding_batch_size),
                        **({"prompt": prefix} if prefix else {}),
                    ).tolist()
                ),
                query,
                prefix,
            )

        return async_embedding_function
    elif embedding_engine in ["ollama", "openai", "azure_openai"]:
        embedding_function = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        async def async_embedding_function(query, prefix=None, user=None):
            if isinstance(query, list):
                # Create batches
                batches = [
                    query[i : i + embedding_batch_size]
                    for i in range(0, len(query), embedding_batch_size)
                ]

                if enable_async:
                    log.debug(
                        f"generate_multiple_async: Processing {len(batches)} batches in parallel"
                    )
                    # Execute all batches in parallel
                    tasks = [
                        embedding_function(batch, prefix=prefix, user=user)
                        for batch in batches
                    ]
                    batch_results = await asyncio.gather(*tasks)
                else:
                    log.debug(
                        f"generate_multiple_async: Processing {len(batches)} batches sequentially"
                    )
                    batch_results = []
                    for batch in batches:
                        batch_results.append(
                            await embedding_function(batch, prefix=prefix, user=user)
                        )

                # Flatten results
                embeddings = []
                for batch_embeddings in batch_results:
                    if isinstance(batch_embeddings, list):
                        embeddings.extend(batch_embeddings)

                log.debug(
                    f"generate_multiple_async: Generated {len(embeddings)} embeddings from {len(batches)} parallel batches"
                )
                return embeddings
            else:
                return await embedding_function(query, prefix, user)

        return async_embedding_function
    elif embedding_engine == "sentence_transformers_cuda":
        # Previously `== "sentence_transformers"`, which was UNREACHABLE (the first
        # branch already matches that value) so this CUDA-stream loader never ran
        # (Phase 1, C-1). Now opt-in via "sentence_transformers_cuda"; existing
        # configs are unchanged. Use only on a GPU host with the model available locally.
        import torch

        if not hasattr(get_embedding_function, "_st_cache"):
            get_embedding_function._st_cache = {}

        cache_key = f"{embedding_model}_cuda"

        if cache_key not in get_embedding_function._st_cache:
            log.info(f"[Embeddings] Loading {embedding_model} on CUDA...")
            _st_model = SentenceTransformer(
                embedding_model,
                device="cuda",
                local_files_only=True,   # ← ADD THIS — never try to download
            )
            _st_model = _st_model.cuda()
            _st_model._cuda_stream = torch.cuda.Stream()
            with torch.cuda.stream(_st_model._cuda_stream):
                _st_model.encode(
                    ["warmup"],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            torch.cuda.synchronize()
            get_embedding_function._st_cache[cache_key] = _st_model
            log.info(f"[Embeddings] Loaded on {next(_st_model.parameters()).device} ✅")
        else:
            _st_model = get_embedding_function._st_cache[cache_key]

        def _encode_on_gpu(texts, batch_size):
            with torch.cuda.stream(_st_model._cuda_stream):
                result = _st_model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            # Sync only the dedicated stream, not the entire device
            _st_model._cuda_stream.synchronize()
            torch.cuda.empty_cache()
            return result.tolist()

        async def async_st_embedding(query, prefix=None, user=None):
            return await asyncio.to_thread(
                _encode_on_gpu,
                query,
                int(embedding_batch_size),
            )

        return async_st_embedding
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


async def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        embeddings = await agenerate_ollama_batch_embeddings(
            **{
                "model": model,
                "texts": text if isinstance(text, list) else [text],
                "url": url,
                "key": key,
                "prefix": prefix,
                "user": user,
            }
        )
        if embeddings is None:
            raise RuntimeError(f"{engine} embedding request returned no data")
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        embeddings = await agenerate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        if embeddings is None:
            raise RuntimeError(f"{engine} embedding request returned no data")
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = await agenerate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        if embeddings is None:
            raise RuntimeError(f"{engine} embedding request returned no data")
        return embeddings[0] if isinstance(text, str) else embeddings


def get_reranking_function(reranking_engine, reranking_model, reranking_function):
    if reranking_function is None:
        return None
    if reranking_engine == "external":
        return lambda query, documents, user=None: reranking_function.predict(
            [(query, doc.page_content) for doc in documents], user=user
        )
    else:
        return lambda query, documents, user=None: reranking_function.predict(
            [(query, doc.page_content) for doc in documents]
        )


# ── Query Intent Router ────────────────────────────────────────────────────────

def _classify_query_route(query: str) -> str:
    """
    Classify query intent to select execution route.
    Returns "metadata", "lightweight", or "full" (default — when in doubt).
    Uses regex + heuristics only; no LLM call.
    """
    _META_RE = re.compile(
        r'\b(how many|count of|list all|all topics|all chapters|'
        r'all subjects|table of contents|show all|enumerate)\b',
        re.IGNORECASE,
    )
    if _META_RE.search(query):
        return "metadata"

    _VERB_RE = re.compile(
        r'\b(is|are|was|were|how|why|what|when|where|'
        r'explain|compare|define|describe|and|or)\b',
        re.IGNORECASE,
    )
    _words = query.split()
    if len(_words) <= 3 and not _VERB_RE.search(query):
        return "lightweight"

    return "full"


def _quick_extract_collection_names(items: list) -> list[str]:
    """Lightweight collection-name extractor for Routes 1 and 2 (no full items loop)."""
    names = []
    for item in items:
        if item.get("collection_name"):
            names.append(item["collection_name"])
        elif item.get("collection_names"):
            names.extend(item["collection_names"])
        elif item.get("type") == "file" and item.get("id") and not item.get("context"):
            raw = f"file-{item['id']}"
            cname = re.sub(r"[^a-zA-Z0-9_]", "", raw.replace("-", "_")).strip("_")
            if cname:
                cname = ("C" + cname) if not cname[0].isalpha() else (cname[0].upper() + cname[1:])
                names.append(cname)
        elif item.get("type") == "collection" and item.get("id") and not item.get("context"):
            names.append(item["id"])
    return names


def _metadata_route(collection_names: list[str], query: str) -> Optional[list]:
    """Route 1: fetch_objects only — no vector, no BM25, no cross-encoder.
    Returns a synthetic structure-summary document from Weaviate metadata fields."""
    all_subjects: set = set()
    all_chapters: set = set()
    all_topics:   set = set()

    for cname in collection_names[:3]:
        try:
            result = VECTOR_DB_CLIENT.query(cname, filter={}, limit=200)
            if result and result.metadatas:
                for m in result.metadatas[0]:
                    if not isinstance(m, dict):
                        continue
                    if m.get("subject"):
                        all_subjects.add(str(m["subject"]))
                    if m.get("chapter"):
                        all_chapters.add(str(m["chapter"]))
                    if m.get("topics"):
                        _t = m["topics"]
                        for t in (_t if isinstance(_t, list) else [_t]):
                            all_topics.add(str(t))
        except Exception as _e:
            log.debug("[Router] metadata_route fetch failed for %s: %s", cname, _e)

    if not any([all_subjects, all_chapters, all_topics]):
        return None

    lines = ["[Document Structure]"]
    if all_subjects:
        lines.append(f"Subjects: {', '.join(sorted(all_subjects))}")
    if all_chapters:
        lines.append(f"Chapters ({len(all_chapters)}): {', '.join(sorted(all_chapters)[:40])}")
    if all_topics:
        lines.append(f"Topics ({len(all_topics)}): {', '.join(sorted(all_topics)[:40])}")

    return [{
        "source": {"type": "merged"},
        "document": ["\n".join(lines)],
        "metadata": [{"source": "metadata_route", "retrieval": "metadata_only"}],
    }]


async def _lightweight_route(
    collection_names: list[str], query: str, k: int
) -> Optional[list]:
    """Route 2: BM25-only — no vector embedding, no cross-encoder.
    Returns top-k BM25 results directly."""
    docs:  list = []
    metas: list = []

    for cname in collection_names[:3]:
        try:
            retriever = WeaviateBM25Retriever(
                collection_name=cname,
                top_k=k,
                query_properties=["raw_text", "keywords"],
            )
            results = await retriever.ainvoke(query)
            for d in results:
                docs.append(d.page_content)
                metas.append(d.metadata)
        except Exception as _e:
            log.debug("[Router] lightweight_route BM25 failed for %s: %s", cname, _e)

    if not docs:
        return None

    return [{
        "source": {"type": "merged"},
        "document": docs[:k],
        "metadata": metas[:k],
    }]


async def get_sources_from_items(
    request,
    items,
    queries,
    embedding_function,
    k,
    reranking_function,
    k_reranker,
    r,
    hybrid_bm25_weight,
    hybrid_search,
    full_context=False,
    user: Optional[UserModel] = None,
):
    log.debug(
        f"items: {items} {queries} {embedding_function} {reranking_function} {full_context}"
    )
    import time as _qt
    _qs_t0 = _qt.monotonic()
    print("", flush=True)
    print("┌" + "─" * 60 + "┐", flush=True)
    print("│  RETRIEVAL  START" + " " * 42 + "│", flush=True)
    print(f"│  Queries : {len(queries):<49}│", flush=True)
    for _qi, _q in enumerate(queries):
        print(f"│  [{_qi+1}] {str(_q)[:56]:<56}│", flush=True)
    print("└" + "─" * 60 + "┘", flush=True)

    _t0_total = time.perf_counter()

    # === RAG RESULT CACHE — lookup ===
    # Default OFF (ENABLE_RAG_RESULT_CACHE). Fail-open: any error falls through
    # to the normal pipeline. Only collection-backed items are cacheable —
    # text/note/chat/url attachments carry content the key can't capture.
    _rc_key_parts = None
    _rc_collections = []
    try:
        from futureverse_ai.retrieval import rag_cache as _rag_cache
        if _rag_cache.RESULT_CACHE_ENABLED and not full_context and items:
            _rc_cacheable = not getattr(
                getattr(request.app.state, "config", None),
                "BYPASS_EMBEDDING_AND_RETRIEVAL", False,
            ) and all(
                item.get("collection_name")
                or item.get("collection_names")
                or (
                    item.get("type") in ("file", "collection")
                    and item.get("id")
                    and not item.get("context")
                    and not item.get("docs")
                )
                for item in items
            )
            if _rc_cacheable:
                _rc_collections = sorted(_quick_extract_collection_names(items))
                _rc_key_parts = [
                    _rc_collections,
                    list(queries),
                    k,
                    bool(hybrid_search),
                    bool(getattr(
                        getattr(request.app.state, "config", None),
                        "ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS", False,
                    )),
                    _rag_cache.EMBEDDING_MODEL,
                    bool(getattr(
                        getattr(request.app.state, "config", None),
                        "ENABLE_NEO4J", False,
                    )),
                ]
                _rc_hit = _rag_cache.cache_get("result", _rc_key_parts)
                if _rc_hit is not None:
                    log.debug(
                        "[RAG result cache] HIT — %d collections, %d queries",
                        len(_rc_collections), len(queries),
                    )
                    return _rc_hit
                log.debug("[RAG result cache] MISS")
    except Exception as _rc_e:
        log.debug("[RAG result cache] lookup failed (fail-open): %s", _rc_e)
        _rc_key_parts = None
    # === END RAG RESULT CACHE lookup ===

    # === QUERY ROUTING ===
    _primary_q = queries[0] if queries else ""
    _route = _classify_query_route(_primary_q)
    log.info("[Router] route=%s query=%r", _route, _primary_q[:80])

    if _route in ("metadata", "lightweight"):
        _cnames = _quick_extract_collection_names(items)
        if _cnames:
            try:
                if _route == "metadata":
                    _route_sources = _metadata_route(_cnames, _primary_q)
                else:
                    _route_sources = await _lightweight_route(_cnames, _primary_q, k)
                if _route_sources:
                    log.info(
                        "[Router] %s route returned %d docs — skipping full pipeline",
                        _route, len(_route_sources[0].get("document", [])),
                    )
                    return _route_sources
            except Exception as _re:
                log.warning(
                    "[Router] %s route failed, falling through to full pipeline: %s",
                    _route, _re,
                )
    # === END QUERY ROUTING ===

    # === QUERY EXPANSION ===
    # Keep the LLM-generated queries from middleware (already 3-4 diverse ones).
    # Add at most one keyword-distilled variant per original query to help BM25,
    # but never add hardcoded topic templates (they generate garbage for non-forest queries).
    _t0 = time.perf_counter()
    try:
        import re
        expanded = list(queries)

        STOP = {'which', 'what', 'where', 'when', 'how', 'this', 'that', 'from',
                'with', 'have', 'been', 'were', 'they', 'their', 'about', 'also',
                'following', 'given', 'above', 'consider', 'statements',
                'correct', 'true', 'false', 'answer', 'option',
                'characteristics', 'description', 'difference', 'between',
                'explain', 'define', 'mention', 'state', 'discuss', 'write'}

        for q in list(queries)[:2]:
            # Strip MCQ option labels: "(a) word" etc.
            clean = re.sub(r'\([a-d]\)\s*[\w\s]{0,30}', ' ', q)
            # Extract meaningful content words (4+ chars, not stopwords)
            words = [w.lower() for w in re.findall(r'[A-Za-z]{4,}', clean)
                     if w.lower() not in STOP]
            if len(words) >= 2:
                # Build a compact keyword query — better BM25 signal than full sentence
                keyword_q = ' '.join(words[:4])
                if keyword_q not in expanded:
                    expanded.append(keyword_q)

        queries = expanded[:6]
        log.info(f"Expanded queries: {queries}")
    except Exception as e:
        log.warning(f"Query expansion failed: {e}")
    log.info("[PERF] QueryExpansion: %.0fms", (time.perf_counter() - _t0) * 1000)

    # Issue 6: cosine-similarity query dedup (replaces Jaccard word-overlap).
    # Semantically identical queries with different vocabulary slip through exact-match
    # dedup and triple retrieval + embedding cost. Embed all variants in one batch call
    # then compute an (N×N) similarity matrix with numpy — ~100× faster than a
    # pure-Python inner loop on 768/1536-dim vectors.
    # Threshold 0.95: only near-perfect paraphrases are dropped; diverse expansions kept.
    _t0 = time.perf_counter()
    if len(queries) > 1:
        try:
            import numpy as _np
            _q_embeds = await embedding_function(queries, RAG_EMBEDDING_QUERY_PREFIX)
            if isinstance(_q_embeds, list) and len(_q_embeds) == len(queries):
                _arr    = _np.array(_q_embeds, dtype=_np.float32)      # (N, D)
                _norms  = _np.linalg.norm(_arr, axis=1, keepdims=True)  # (N, 1)
                _arr_n  = _arr / (_norms + 1e-9)                        # unit vectors
                _sim_mx = _arr_n @ _arr_n.T                             # (N, N) cosine sims

                _keep = [True] * len(queries)
                for _qi in range(1, len(queries)):
                    for _qj in range(_qi):
                        if _keep[_qj] and float(_sim_mx[_qi, _qj]) > 0.95:
                            _keep[_qi] = False
                            log.debug(
                                "[QueryDedup] Dropped '%s' (cos=%.3f with '%s')",
                                queries[_qi], float(_sim_mx[_qi, _qj]), queries[_qj],
                            )
                            break

                _kept_q = [q for q, k in zip(queries, _keep) if k]
                if len(_kept_q) < len(queries):
                    log.info(
                        "[QueryDedup] %d → %d queries after cosine dedup (dropped %d semantic duplicates)",
                        len(queries), len(_kept_q), len(queries) - len(_kept_q),
                    )
                queries = _kept_q
        except Exception as _qd_e:
            log.debug("[QueryDedup] Cosine dedup skipped (non-fatal): %s", _qd_e)
    log.info("[PERF] CosineDedup: %.0fms", (time.perf_counter() - _t0) * 1000)


    extracted_collections = []
    query_results = []
    _hs_idx = 0

    for item in items:
        query_result = None
        collection_names = []

        if item.get("type") == "text":
            # Raw Text
            # Used during temporary chat file uploads or web page & youtube attachements

            if item.get("context") == "full":
                if item.get("file"):
                    # if item has file data, use it
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content")]
                        ],
                        "metadatas": [[item.get("file", {}).get("meta", {})]],
                    }

            if query_result is None:
                # Fallback
                if item.get("collection_name"):
                    # If item has a collection name, use it
                    collection_names.append(item.get("collection_name"))
                elif item.get("file"):
                    # If item has file data, use it
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content")]
                        ],
                        "metadatas": [[item.get("file", {}).get("meta", {})]],
                    }
                else:
                    # Fallback to item content
                    query_result = {
                        "documents": [[item.get("content")]],
                        "metadatas": [
                            [{"file_id": item.get("id"), "name": item.get("name")}]
                        ],
                    }

        elif item.get("type") == "note":
            # Note Attached
            note = Notes.get_note_by_id(item.get("id"))

            if note and (
                user.role == "admin"
                or note.user_id == user.id
                or AccessGrants.has_access(
                    user_id=user.id,
                    resource_type="note",
                    resource_id=note.id,
                    permission="read",
                )
            ):
                # User has access to the note
                query_result = {
                    "documents": [[note.data.get("content", {}).get("md", "")]],
                    "metadatas": [[{"file_id": note.id, "name": note.title}]],
                }

        elif item.get("type") == "chat":
            # Chat Attached
            chat = Chats.get_chat_by_id(item.get("id"))

            if chat and (user.role == "admin" or chat.user_id == user.id):
                messages_map = chat.chat.get("history", {}).get("messages", {})
                message_id = chat.chat.get("history", {}).get("currentId")

                if messages_map and message_id:
                    # Reconstruct the message list in order
                    message_list = get_message_list(messages_map, message_id)
                    message_history = "\n".join(
                        [
                            f"#### {m.get('role', 'user').capitalize()}\n{m.get('content')}\n"
                            for m in message_list
                        ]
                    )

                    # User has access to the chat
                    query_result = {
                        "documents": [[message_history]],
                        "metadatas": [[{"file_id": chat.id, "name": chat.title}]],
                    }

        elif item.get("type") == "url":
            content, docs = get_content_from_url(request, item.get("url"))
            if docs:
                query_result = {
                    "documents": [[content]],
                    "metadatas": [[{"url": item.get("url"), "name": item.get("url")}]],
                }
        elif item.get("type") == "file":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                if item.get("file", {}).get("data", {}).get("content", ""):
                    # Manual Full Mode Toggle
                    # Used from chat file modal, we can assume that the file content will be available from item.get("file").get("data", {}).get("content")
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content", "")]
                        ],
                        "metadatas": [
                            [
                                {
                                    "file_id": item.get("id"),
                                    "name": item.get("name"),
                                    **item.get("file")
                                    .get("data", {})
                                    .get("metadata", {}),
                                }
                            ]
                        ],
                    }
                elif item.get("id"):
                    file_object = Files.get_file_by_id(item.get("id"))
                    if file_object:
                        query_result = {
                            "documents": [[file_object.data.get("content", "")]],
                            "metadatas": [
                                [
                                    {
                                        "file_id": item.get("id"),
                                        "name": file_object.filename,
                                        "source": file_object.filename,
                                    }
                                ]
                            ],
                        }
            else:
                # Fallback to collection names
                if item.get("legacy"):
                    collection_names.append(f"{item['id']}")
                else:
                    collection_names.append(f"file-{item['id']}")

        elif item.get("type") == "collection":
            # Manual Full Mode Toggle for Collection
            knowledge_base = Knowledges.get_knowledge_by_id(item.get("id"))

            if knowledge_base and (
                user.role == "admin"
                or knowledge_base.user_id == user.id
                or AccessGrants.has_access(
                    user_id=user.id,
                    resource_type="knowledge",
                    resource_id=knowledge_base.id,
                    permission="read",
                )
            ):
                if (
                    item.get("context") == "full"
                    or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
                ):
                    if knowledge_base and (
                        user.role == "admin"
                        or knowledge_base.user_id == user.id
                        or AccessGrants.has_access(
                            user_id=user.id,
                            resource_type="knowledge",
                            resource_id=knowledge_base.id,
                            permission="read",
                        )
                    ):
                        files = Knowledges.get_files_by_id(knowledge_base.id)

                        documents = []
                        metadatas = []
                        for file in files:
                            documents.append(file.data.get("content", ""))
                            metadatas.append(
                                {
                                    "file_id": file.id,
                                    "name": file.filename,
                                    "source": file.filename,
                                }
                            )

                        query_result = {
                            "documents": [documents],
                            "metadatas": [metadatas],
                        }
                else:
                    # Fallback to collection names
                    if item.get("legacy"):
                        collection_names = item.get("collection_names", [])
                    else:
                        collection_names.append(item["id"])
                        # Also add all file collections belonging to this KB
                        try:
                            kb_files = Knowledges.get_files_by_id(item["id"])
                            for kf in kb_files:
                                # Mirror WeaviateClient._sanitize_collection_name logic
                                raw = f"file-{kf.id}"
                                file_col = re.sub(r"[^a-zA-Z0-9_]", "", raw.replace("-", "_")).strip("_")
                                if file_col:
                                    if not file_col[0].isalpha():
                                        file_col = "C" + file_col
                                    else:
                                        file_col = file_col[0].upper() + file_col[1:]
                                    if file_col not in collection_names:
                                        collection_names.append(file_col)
                                        log.info(f"KB file collection added: {file_col}")
                        except Exception as _kbe:
                            log.warning(f"KB file collection lookup failed: {_kbe}")

        elif item.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            query_result = {
                "documents": [[doc.get("content") for doc in item.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in item.get("docs")]],
            }
        elif item.get("collection_name"):
            # Direct Collection Name
            collection_names.append(item["collection_name"])
        elif item.get("collection_names"):
            # Collection Names List
            collection_names.extend(item["collection_names"])

        # If query_result is None
        # Fallback to collection names and vector search the collections
        if query_result is None and collection_names:
            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                log.debug(f"skipping {item} as it has already been extracted")
                continue

            try:
                if full_context:
                    query_result = get_all_items_from_collections(collection_names)
                else:
                    query_result = None  # Initialize to None
                    if hybrid_search:
                        try:
                            _hs_idx += 1
                            _t0 = time.perf_counter()
                            query_result = await query_collection_with_hybrid_search(
                                collection_names=collection_names,
                                queries=queries,
                                embedding_function=embedding_function,
                                k=k,
                                reranking_function=reranking_function,
                                k_reranker=k_reranker,
                                r=r,
                                hybrid_bm25_weight=hybrid_bm25_weight,
                                enable_enriched_texts=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS,
                            )
                            log.info("[PERF] HybridSearch_Q%d: %.0fms", _hs_idx, (time.perf_counter() - _t0) * 1000)
                        except Exception as e:
                            log.warning(
                                f"Hybrid search failed ({e}); falling back to vector-only search."
                            )

                    # fallback to vector-only search whenever hybrid_search failed OR is disabled
                    if query_result is None:
                        query_result = await query_collection(
                            collection_names=collection_names,
                            queries=queries,
                            embedding_function=embedding_function,
                            k=k,
                        )
            except Exception as e:
                log.exception(e)

            extracted_collections.extend(collection_names)

        if query_result:
            if "data" in item:
                del item["data"]
            query_results.append({**query_result, "file": item})

    # === KNOWLEDGE GRAPH INJECTION ===
    if getattr(getattr(request.app.state, "config", None), "ENABLE_NEO4J", False):
        try:
            from futureverse_ai.retrieval.graph.graph_rag import get_graph_source

            # Build metadata_context from already-retrieved chunks so graph search
            # can rank by subject / chapter matching the active knowledge base.
            _meta_ctx: dict = {}
            for qr in query_results:
                metas = qr.get("metadatas", [[]])[0]
                for m in (metas or []):
                    if isinstance(m, dict):
                        if not _meta_ctx.get("subject") and m.get("subject"):
                            _meta_ctx["subject"] = m["subject"]
                        if not _meta_ctx.get("book") and m.get("book"):
                            _meta_ctx["book"] = m["book"]
                        if not _meta_ctx.get("chapter") and m.get("chapter"):
                            _meta_ctx["chapter"] = m["chapter"]
                        if all(_meta_ctx.get(k) for k in ("subject", "book", "chapter")):
                            break
                if all(_meta_ctx.get(k) for k in ("subject", "book", "chapter")):
                    break

            graph_result = await asyncio.to_thread(
                get_graph_source,
                queries=queries,
                limit=10,
                metadata_context=_meta_ctx or None,
            )
            if graph_result:
                query_results.append({
                    **graph_result,
                    "file": {
                        "type": "graph",
                        "id": "knowledge_graph",
                        "name": "Knowledge Graph",
                        "collection_name": "knowledge_graph",
                    }
                })
                log.info(
                    f"Knowledge graph context injected "
                    f"(subject={_meta_ctx.get('subject','?')}, "
                    f"chapter={_meta_ctx.get('chapter','?')})"
                )
        except Exception as e:
            log.warning(f"Knowledge graph injection failed: {e}")

    # === END KNOWLEDGE GRAPH INJECTION ===

    # === GRAPH-GUIDED VECTOR RE-RANKING ======================================
    if getattr(getattr(request.app.state, "config", None), "ENABLE_NEO4J", False):
        try:
            _graph_entity_names: list[str] = []
            for _qr in query_results:
                if _qr.get("file", {}).get("type") == "graph":
                    _graph_entity_names = _qr.get("entity_names", [])
                    break

            if _graph_entity_names:
                _entity_anchors = [e.lower() for e in _graph_entity_names if e]

                def _entity_score(text: str) -> int:
                    tl = text.lower()
                    return sum(1 for et in _entity_anchors if et in tl)

                _reranked = 0
                for _qr in query_results:
                    if _qr.get("file", {}).get("type") == "graph":
                        continue
                    _docs  = _qr.get("documents",  [[]])[0]
                    _metas = _qr.get("metadatas",  [[]])[0]
                    _dists = (_qr.get("distances", [[]])[0]
                              if _qr.get("distances") else [])

                    if len(_docs) <= 1:
                        continue

                    _scored = sorted(
                        range(len(_docs)),
                        key=lambda _i: (
                            -_entity_score(
                                _docs[_i] if isinstance(_docs[_i], str) else ""
                            ),
                            _dists[_i] if _i < len(_dists) else 1.0,
                        ),
                    )
                    if _scored == list(range(len(_docs))):
                        continue

                    _qr["documents"] = [[_docs[_i] for _i in _scored]]
                    _qr["metadatas"] = [
                        [_metas[_i] if _i < len(_metas) else {} for _i in _scored]
                    ]
                    if _dists:
                        _qr["distances"] = [
                            [_dists[_i] if _i < len(_dists) else 1.0
                             for _i in _scored]
                        ]
                    _reranked += 1

                if _reranked:
                    log.info(
                        f"Graph-guided re-ranking: {len(_entity_anchors)} entity "
                        f"anchors applied to {_reranked} vector collections"
                    )
        except Exception as _rr_exc:
            log.warning(f"Graph-guided re-ranking failed (non-fatal): {_rr_exc}")
    # === END GRAPH-GUIDED VECTOR RE-RANKING ===================================

    sources = []
    for query_result in query_results:
        try:
            if "documents" in query_result:
                if "metadatas" in query_result:
                    source = {
                        "source": query_result["file"],
                        "document": query_result["documents"][0],
                        "metadata": query_result["metadatas"][0],
                    }
                    if "distances" in query_result and query_result["distances"]:
                        source["distances"] = query_result["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)
    # === UPSC CONTEXT SIGNAL ===
    try:
        if queries:
            upsc_signal = {
                "source": {"type": "system"},
                "document": [
                    "UPSC Context: Provide analytical explanations, causes, impacts, examples, and conceptual clarity suitable for UPSC Prelims and Mains."
                ],
                "metadata": [{"type": "upsc_signal"}],
            }

            sources.insert(0, upsc_signal)

    except Exception as e:
        log.warning(f"UPSC signal injection failed: {e}")

    try:
        import os
        from datetime import datetime
        _debug_on = os.environ.get("RAG_DEBUG_LOG", "false").lower() == "true"
        # Was a hardcoded absolute Windows path that broke on Linux/Docker and
        # wrote on every query; now portable + opt-in (RAG_DEBUG_LOG[_DIR]).
        log_dir = os.environ.get("RAG_DEBUG_LOG_DIR", os.path.join(os.getcwd(), "logs"))
        if _debug_on:
            os.makedirs(log_dir, exist_ok=True)
        log_path    = os.path.join(log_dir, "retrieval_debug.log")
        _log_queries = queries
        _log_sources = len(query_results)
        _log_hybrid  = hybrid_search
        _log_items   = [
            (qr.get("file", {}),
             list(qr.get("documents", [[]])[0]),
             list(qr.get("metadatas",  [[]])[0]))
            for qr in query_results
        ]

        def _write_debug_log():
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("\n" + "="*80 + "\n")
                lf.write(f"TIME   : {ts}\n")
                lf.write(f"QUERIES: {_log_queries}\n")
                lf.write(f"SOURCES: {_log_sources}\n")
                lf.write(f"HYBRID : {_log_hybrid}\n")
                lf.write("-"*80 + "\n")
                for i, (fi, docs, metas) in enumerate(_log_items):
                    lf.write(f"\n[{i+1}] {fi.get('name','?')} | chunks={len(docs)}\\n")
                    for j, (doc, meta) in enumerate(zip(docs, metas)):
                        m = meta if isinstance(meta, dict) else {}
                        lf.write(f"  Chunk {j+1} | {m.get('source','')} | {m.get('chapter','')}\\n")
                        lf.write(f"  Via: {m.get('retrieval','weaviate')}\\n")
                        lf.write(f"  Text: {str(doc)[:400].replace(chr(10),' ')}...\\n")
                lf.write("="*80 + "\\n")

        if _debug_on:
            await asyncio.to_thread(_write_debug_log)
    except Exception as e:
        log.warning(f"Retrieval debug log failed: {e}")

    # === DEDUPLICATE CONTEXT ===
    # Keep the copy with the highest reranker score (metadata["score"]).
    # Last-write-wins was wrong: multiple sources may surface the same parent_text
    # with different scores; the highest-scoring copy should be the survivor.
    _t0_merge = time.perf_counter()
    unique_docs = {}
    for s in sources:
        for i, doc in enumerate(s["document"]):
            key = hashlib.sha256(doc.encode()).hexdigest()
            _meta_i = s["metadata"][i]
            _incoming = float(_meta_i.get("score", 0.0)) if isinstance(_meta_i, dict) else 0.0
            if key not in unique_docs:
                unique_docs[key] = (doc, _meta_i)
            else:
                _existing = float(unique_docs[key][1].get("score", 0.0)) if isinstance(unique_docs[key][1], dict) else 0.0
                if _incoming > _existing:
                    unique_docs[key] = (doc, _meta_i)

    sources = [{
        "source": {"type": "merged"},
        "document": [v[0] for v in unique_docs.values()],
        "metadata": [v[1] for v in unique_docs.values()],
    }]

    # === POST-MERGE SEMANTIC DEDUP + RERANK ===
    # Runs after SHA-256 dedup so we work on the already-exact-deduped set.
    # KG block and UPSC signal are excluded from dedup/rerank (kept as-is at top).
    try:
        _primary_query = queries[0] if queries else ""
        if _primary_query and sources:
            _all_docs  = sources[0]["document"]
            _all_metas = sources[0]["metadata"]

            # Partition: special blocks vs rerank-eligible vector chunks
            _special_idx: list[int] = []
            _vec_idx:     list[int] = []
            for _i, _m in enumerate(_all_metas):
                _mtype = (_m.get("type", "") if isinstance(_m, dict) else "")
                if _mtype in ("graph", "upsc_signal"):
                    _special_idx.append(_i)
                else:
                    _vec_idx.append(_i)

            # Jaccard semantic dedup — skip words ≤3 chars (stopwords inflate score)
            _seen_sigs:   list       = []
            _deduped_vec: list[int]  = []
            for _i in _vec_idx:
                _text  = str(_all_docs[_i])
                _words = frozenset(w for w in _text.lower().split() if len(w) > 3)
                _is_dup = any(
                    len(_words & _s) / max(len(_words | _s), 1) > 0.75
                    for _s in _seen_sigs if _s
                )
                if not _is_dup:
                    _seen_sigs.append(_words)
                    _deduped_vec.append(_i)
            log.info("[PERF] MergeDedup: %.0fms", (time.perf_counter() - _t0_merge) * 1000)

            # Post-merge rerank with the already-loaded bge-reranker-v2-m3.
            # Bypass when the merged pool came from a single collection and ≤2
            # query variants: Stage 1 already produced calibrated sigmoid scores
            # for all candidates, so a second cross-encoder pass adds latency
            # without changing rank order in the single-source case.
            _real_source_count = sum(
                1 for s in sources
                if not any(
                    isinstance(m, dict) and m.get("retrieval_failed")
                    for m in s.get("metadata", [])
                )
            )
            _query_count = len(queries) if isinstance(queries, list) else 1

            if _real_source_count <= 1 and _query_count <= 2:
                log.debug(
                    "[GlobalRerank] Skipped — single source, %d queries. "
                    "Stage 1 scores already sufficient.",
                    _query_count,
                )
            elif reranking_function and len(_deduped_vec) > 1:
                try:
                    _rtexts = [str(_all_docs[_i]) for _i in _deduped_vec]
                    _s2_t0 = time.perf_counter()
                    if hasattr(reranking_function, "predict"):
                        # Raw CrossEncoder / BGE model — offload to thread pool
                        _pairs   = [(_primary_query, t) for t in _rtexts]
                        _rscores = await asyncio.to_thread(
                            reranking_function.predict, _pairs
                        )
                    else:
                        # Wrapped lambda expecting Document objects
                        _ldocs   = [Document(page_content=t) for t in _rtexts]
                        _rscores = await asyncio.to_thread(
                            reranking_function, _primary_query, _ldocs
                        )
                    log.info(
                        "[PERF] GlobalRerank: %.0fms",
                        (time.perf_counter() - _s2_t0) * 1000,
                    )

                    if _rscores is not None and len(_rscores) == len(_deduped_vec):
                        _ranked = sorted(
                            zip(_rscores, _deduped_vec),
                            key=lambda x: x[0], reverse=True,
                        )
                        _deduped_vec = [_idx for _, _idx in _ranked]
                        log.info(
                            f"Post-merge rerank: top={_ranked[0][0]:.3f} "
                            f"bot={_ranked[-1][0]:.3f} "
                            f"n={len(_deduped_vec)}"
                        )
                except Exception as _rr_e:
                    log.debug(f"Post-merge rerank skipped: {_rr_e}")

            # Rebuild: special blocks first, then reranked/deduped vector chunks
            _final_idx = _special_idx + _deduped_vec
            sources = [{
                "source": {"type": "merged"},
                "document": [_all_docs[_i] for _i in _final_idx],
                "metadata": [_all_metas[_i] for _i in _final_idx],
            }]
            log.info(
                f"Post-merge pipeline: {len(_all_docs)} → {len(_final_idx)} "
                f"(special={len(_special_idx)} KG+signal, "
                f"vec_raw={len(_vec_idx)}, vec_final={len(_deduped_vec)})"
            )
    except Exception as _pm_exc:
        log.warning(f"Post-merge dedup/rerank failed (non-fatal): {_pm_exc}")
    log.info("[PERF] TotalRetrieval: %.0fms", (time.perf_counter() - _t0_total) * 1000)
    # === END POST-MERGE ===

    # ── Retrieval summary to stdout ───────────────────────────────────────────
    try:
        import time as _qt
        _qs_ms    = (_qt.monotonic() - _qs_t0) * 1000
        _n_unique = len(sources[0]["document"]) if sources else 0
        _n_raw    = sum(len(qr.get("documents", [[]])[0]) for qr in query_results)
        print("", flush=True)
        print("┌" + "─" * 60 + "┐", flush=True)
        print("│  RETRIEVAL  COMPLETE" + " " * 39 + "│", flush=True)
        print(f"│  Queries     : {len(queries):<44}│", flush=True)
        print(f"│  Raw chunks  : {_n_raw:<44}│", flush=True)
        print(f"│  Final docs  : {_n_unique:<44}│", flush=True)
        print(f"│  Wall time   : {_qs_ms:.0f}ms{'':<41}│", flush=True)
        print("│" + " " * 60 + "│", flush=True)
        for _qi, _q in enumerate(queries):
            print(f"│  Q{_qi+1}: {str(_q)[:55]:<55}│", flush=True)
        print("│" + " " * 60 + "│", flush=True)
        for _si, _src in enumerate(query_results):
            _fi   = _src.get("file", {})
            _docs = _src.get("documents", [[]])[0]
            _name = _fi.get("name", _fi.get("id", "?"))[:38]
            print(f"│  src[{_si+1}] {_name:<38}  chunks={len(_docs):>3}  │", flush=True)
            for _di, _doc in enumerate(_docs[:3]):   # show first 3 chunks preview
                _prev = str(_doc)[:50].replace("\n", " ")
                print(f"│       chunk{_di+1}: {_prev:<50}│", flush=True)
            if len(_docs) > 3:
                print(f"│       ... {len(_docs)-3} more chunks{'':<35}│", flush=True)
        print("└" + "─" * 60 + "┘", flush=True)
        print("", flush=True)
    except Exception as _rlog_err:
        log.debug("Retrieval summary print failed: %s", _rlog_err)

    # === RAG RESULT CACHE — store ===
    # Never cache errors, empty results, or the "__NO_RESULTS__" sentinel.
    if _rc_key_parts is not None:
        try:
            from futureverse_ai.retrieval import rag_cache as _rag_cache
            _rc_docs = [str(d) for s in sources for d in (s.get("document") or [])]
            _rc_has_real = any(
                not (isinstance(m, dict) and m.get("type") == "upsc_signal")
                for s in sources
                for m in (s.get("metadata") or [])
            )
            if _rc_docs and _rc_has_real and "__NO_RESULTS__" not in _rc_docs:
                _rag_cache.cache_set(
                    "result", _rc_key_parts, sources, _rag_cache.RESULT_CACHE_TTL
                )
                _rag_cache.kbidx_register(
                    _rc_collections,
                    _rag_cache.make_key("result", _rc_key_parts),
                    _rag_cache.RESULT_CACHE_TTL,
                )
                log.debug("[RAG result cache] stored %d docs", len(_rc_docs))
        except Exception as _rc_e:
            log.debug("[RAG result cache] store failed (fail-open): %s", _rc_e)
    # === END RAG RESULT CACHE store ===

    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        _callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.

        """
        return []

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        _callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        scores = None
        if reranking:
            # Run reranker on GPU via dedicated thread
            import torch
            def _rerank_on_gpu(query, docs):
                with torch.cuda.device(0):
                    return self.reranking_function(query, docs)

            # === RAG RERANK CACHE — raw cross-encoder scores keyed by
            # (sha256(query), sha256(chunk), reranker model). Default OFF;
            # fail-open: any cache error scores everything as before. Only
            # cache-misses are sent to the model; ordering/threshold logic
            # downstream is unchanged (raw scores are cached pre-sigmoid).
            _rk_parts, _rk_hits = None, None
            try:
                from futureverse_ai.retrieval import rag_cache as _rag_cache
                if _rag_cache.RERANK_CACHE_ENABLED and documents:
                    _rk_qh = hashlib.sha256(query.encode("utf-8")).hexdigest()
                    _rk_parts = [
                        [
                            _rk_qh,
                            hashlib.sha256(
                                (doc.page_content or "").encode("utf-8")
                            ).hexdigest(),
                            _rag_cache.RERANKING_MODEL,
                        ]
                        for doc in documents
                    ]
                    _rk_hits = _rag_cache.cache_get_many("rerank", _rk_parts)
                    log.debug(
                        "[RAG rerank cache] %d/%d hits",
                        sum(1 for v in _rk_hits if v is not None),
                        len(documents),
                    )
            except Exception as _rk_e:
                log.debug("[RAG rerank cache] lookup failed (fail-open): %s", _rk_e)
                _rk_parts, _rk_hits = None, None

            if _rk_hits is not None and all(v is not None for v in _rk_hits):
                scores = [float(v) for v in _rk_hits]
            else:
                _rk_miss_idx = None
                if _rk_hits is not None and any(v is not None for v in _rk_hits):
                    # Partial hit: score only the misses, merge back in order.
                    _rk_miss_idx = [
                        i for i, v in enumerate(_rk_hits) if v is None
                    ]
                    _rk_miss_scores = await asyncio.to_thread(
                        _rerank_on_gpu,
                        query,
                        [documents[i] for i in _rk_miss_idx],
                    )
                    _rk_miss_list = (
                        list(_rk_miss_scores)
                        if isinstance(_rk_miss_scores, list)
                        else _rk_miss_scores.tolist()
                    )
                    if len(_rk_miss_list) == len(_rk_miss_idx):
                        scores = [
                            float(v) if v is not None else 0.0 for v in _rk_hits
                        ]
                        for _i, _s in zip(_rk_miss_idx, _rk_miss_list):
                            scores[_i] = float(_s)
                if scores is None:
                    # Full miss (or partial merge mismatch): score everything.
                    _rk_miss_idx = list(range(len(documents)))
                    scores = await asyncio.to_thread(_rerank_on_gpu, query, documents)
                # Write freshly computed scores back to the cache.
                if _rk_parts is not None and scores is not None:
                    try:
                        from futureverse_ai.retrieval import rag_cache as _rag_cache
                        _rk_all = (
                            list(scores)
                            if isinstance(scores, list)
                            else scores.tolist()
                        )
                        _rag_cache.cache_set_many(
                            "rerank",
                            [
                                (_rk_parts[_i], float(_rk_all[_i]))
                                for _i in _rk_miss_idx
                            ],
                            _rag_cache.RERANK_CACHE_TTL,
                        )
                    except Exception as _rk_e:
                        log.debug(
                            "[RAG rerank cache] store failed (fail-open): %s", _rk_e
                        )
            # === END RAG RERANK CACHE ===
        else:
            query_embedding = await self.embedding_function(
                query, RAG_EMBEDDING_QUERY_PREFIX
            )
            document_embedding = await self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            # Use GPU tensors for cosine similarity when CUDA is available
            try:
                import torch as _torch
                _cuda = _torch.cuda.is_available()
            except Exception:
                _cuda = False
            if _cuda:
                q = _torch.tensor(query_embedding, device="cuda")
                d = _torch.tensor(document_embedding, device="cuda")
                scores = _torch.nn.functional.cosine_similarity(
                    q.unsqueeze(0), d, dim=-1
                ).cpu().tolist()
            else:
                from sentence_transformers import util
                scores = util.cos_sim(query_embedding, document_embedding)[0].tolist()

        if scores is not None:
            raw_scores = scores.tolist() if not isinstance(scores, list) else list(scores)

            # Issue 4: sigmoid-normalize cross-encoder logits → [0, 1] before applying
            # threshold and metadata bonus. bge-reranker-v2-m3 outputs raw logits in
            # ~[-10, +10]; adding a +0.05 metadata bonus to raw logits is noise (<0.5%
            # of scale). After sigmoid, +0.05 is meaningful (~5% of [0,1] range).
            # Cosine-similarity path (reranking=False) is already in [-1,1] — no sigmoid.
            # Log both raw and normalized scores for distribution monitoring.
            if reranking:
                scores_norm = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
                log.debug(
                    "Reranker raw logits  : min=%.3f  max=%.3f  mean=%.3f",
                    min(raw_scores), max(raw_scores),
                    sum(raw_scores) / len(raw_scores) if raw_scores else 0,
                )
                log.debug(
                    "Reranker sigmoid-norm: min=%.3f  max=%.3f  mean=%.3f",
                    min(scores_norm), max(scores_norm),
                    sum(scores_norm) / len(scores_norm) if scores_norm else 0,
                )
            else:
                scores_norm = raw_scores

            docs_with_scores = list(zip(documents, scores_norm))

            # Threshold operates on sigmoid-normalized [0,1] scores when reranking.
            # Default floor 0.3 rejects logit ≤ -0.85 (clearly off-topic chunks).
            # BREAKING CHANGE: RAG_RELEVANCE_THRESHOLD (r_score) is now interpreted
            # on the [0,1] sigmoid scale, not raw logits. Default changed from 0.0 → 0.3.
            # MIGRATION: translate a previous custom threshold with:
            #   new_value = 1 / (1 + exp(-old_value))  (e.g. old 2.0 → new 0.88)
            effective_floor = max(
                self.r_score if self.r_score else 0.0,
                0.3 if reranking else 0.05,
            )
            before = len(docs_with_scores)
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= effective_floor
            ]
            log.debug(
                "Reranker threshold %.2f (%s): kept %d/%d docs",
                effective_floor,
                "sigmoid-norm" if reranking else "cosine",
                len(docs_with_scores), before,
            )

            result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
            final_results = []
            for doc, doc_score in result[: self.top_n]:
                metadata = doc.metadata

                # Metadata bonuses — deliberately calibrated as [0,1]-scale boost signals
                # for NCERT corpus quality indicators. These are NOT noise compensation;
                # after sigmoid normalization each field contributes a meaningful ~3–5%
                # increment. Max combined bonus = 0.05+0.05+0.03+0.04 = +0.17.
                #
                # Design invariant: the max +0.17 bonus cannot rescue a truly irrelevant
                # chunk over the 0.3 threshold on its own. A chunk needs a base sigmoid
                # score ≥ 0.13 (raw logit ≥ -1.95) before metadata can push it over 0.3.
                # Math: 0.13 + 0.17 = 0.30 (just meets threshold). Below 0.13 → no rescue.
                bonus = 0

                if metadata.get("chapter"):
                    bonus += 0.05

                if metadata.get("subject"):
                    bonus += 0.05

                if metadata.get("class_level"):
                    bonus += 0.03

                if metadata.get("title"):
                    bonus += 0.04

                doc_score = doc_score + bonus

                metadata["score"] = doc_score
                doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
                final_results.append(doc)
            return final_results
        else:
            log.warning(
                "No valid scores found, check your reranking function. Returning original documents."
            )
            return documents
