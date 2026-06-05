import logging
import weaviate
import re
import uuid
from typing import Any, Dict, List, Optional, Union

_log = logging.getLogger(__name__)

from futureverse_ai.retrieval.vector.main import (
    VectorDBBase,
    VectorItem,
    SearchResult,
    GetResult,
)
from futureverse_ai.retrieval.vector.utils import process_metadata
from futureverse_ai.config import (
    WEAVIATE_HTTP_HOST,
    WEAVIATE_GRPC_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_API_KEY,
    WEAVIATE_HTTP_SECURE,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_SKIP_INIT_CHECKS,
)


def _build_weaviate_filter(filter_dict: Optional[dict]):
    """Convert {field: value, ...} to a weaviate.classes.query.Filter.
    Multiple keys are ANDed. Returns None for empty/None input.
    Supports operators: {"field": {"$in": [v1, v2, ...]}} → OR filter.
    Shared by search(), bm25_search(), query(), and delete()."""
    if not filter_dict:
        return None
    wf = None
    for key, value in filter_dict.items():
        if isinstance(value, dict) and "$in" in value:
            # Build OR filter: field == v1 OR field == v2 OR ...
            values = value["$in"]
            if not values:
                continue
            or_parts = [
                weaviate.classes.query.Filter.by_property(name=key).equal(v)
                for v in values
            ]
            pf = (
                or_parts[0]
                if len(or_parts) == 1
                else weaviate.classes.query.Filter.any_of(or_parts)
            )
        else:
            pf = weaviate.classes.query.Filter.by_property(name=key).equal(value)
        wf = pf if wf is None else weaviate.classes.query.Filter.all_of([wf, pf])
    return wf


def _convert_uuids_to_strings(obj: Any) -> Any:
    """
    Recursively convert UUID objects to strings in nested data structures.

    This function handles:
    - UUID objects -> string
    - Dictionaries with UUID values
    - Lists/Tuples with UUID values
    - Nested combinations of the above

    Args:
        obj: Any object that might contain UUIDs

    Returns:
        The same object structure with UUIDs converted to strings
    """
    if isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_uuids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_uuids_to_strings(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return obj


class WeaviateClient(VectorDBBase):
    def __init__(self):
        self.url = WEAVIATE_HTTP_HOST
        self._connect()

    def _connect(self) -> None:
        """Create or re-create the Weaviate HTTP/gRPC connection.

        Uses generous timeouts because the pipeline can be idle for 200+ seconds
        during contextual chunking before the first Weaviate write.
        """
        additional_config = weaviate.classes.init.AdditionalConfig(
            timeout=weaviate.classes.init.Timeout(
                init=5,      # 5 s startup health-check — fail fast if Weaviate is down
                query=120,   # 2 min — covers pauses during contextual chunking
                insert=300,  # 5 min — large batch inserts
            )
        )
        try:
            # Build connection parameters
            # Weaviate Cloud (WCS) - use connect_to_weaviate_cloud for .weaviate.cloud hosts
            if "weaviate.cloud" in WEAVIATE_HTTP_HOST:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=f"https://{WEAVIATE_HTTP_HOST}",
                    auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
                    skip_init_checks=WEAVIATE_SKIP_INIT_CHECKS,
                    additional_config=additional_config,
                )
            elif WEAVIATE_HTTP_HOST in ("localhost", "127.0.0.1"):
                self.client = weaviate.connect_to_local(
                    host="localhost",
                    port=WEAVIATE_HTTP_PORT,
                    grpc_port=WEAVIATE_GRPC_PORT,
                    additional_config=additional_config,
                )
            else:
                connection_params = {
                    "http_host": WEAVIATE_HTTP_HOST,
                    "http_port": WEAVIATE_HTTP_PORT,
                    "http_secure": WEAVIATE_HTTP_SECURE,
                    "grpc_host": WEAVIATE_GRPC_HOST,
                    "grpc_port": WEAVIATE_GRPC_PORT,
                    "grpc_secure": WEAVIATE_GRPC_SECURE,
                    "skip_init_checks": WEAVIATE_SKIP_INIT_CHECKS,
                    "additional_config": additional_config,
                }
                if WEAVIATE_API_KEY:
                    connection_params["auth_credentials"] = (
                        weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY)
                    )
                self.client = weaviate.connect_to_custom(**connection_params)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}") from e

    def _reconnect(self) -> None:
        """Close the stale connection and re-establish it.

        Called automatically when a WeaviateTimeoutError is detected on a
        collections.exists() call, which indicates the underlying httpx
        connection pool dropped the socket after a long idle period.
        """
        _log.warning(
            "[Weaviate] Connection appears stale (idle timeout) — reconnecting..."
        )
        try:
            self.client.close()
        except Exception:
            pass
        self._connect()
        _log.info("[Weaviate] Reconnected successfully.")

    def _collections_exists(self, sane_name: str) -> bool:
        """Check whether a collection exists, with one auto-reconnect retry.

        The ingestion pipeline can be idle for 200+ seconds during contextual
        chunking.  The httpx connection pool drops idle sockets well before that,
        so the first collections.exists() call after chunking raises
        WeaviateTimeoutError.  We catch it here, reconnect once, and retry —
        callers never see the error and the pipeline continues normally.
        """
        try:
            return self.client.collections.exists(sane_name)
        except weaviate.exceptions.WeaviateTimeoutError:
            _log.warning(
                "[Weaviate] Timeout on collections.exists('%s') — reconnecting and retrying.",
                sane_name,
            )
            self._reconnect()
            return self.client.collections.exists(sane_name)

    def _sanitize_collection_name(self, collection_name: str) -> str:
        """Sanitize collection name to be a valid Weaviate class name."""
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name must be a non-empty string")

        # Requirements for a valid Weaviate class name:
        # The collection name must begin with a capital letter.
        # The name can only contain letters, numbers, and the underscore (_) character. Spaces are not allowed.

        # Replace hyphens with underscores and keep only alphanumeric characters
        name = re.sub(r"[^a-zA-Z0-9_]", "", collection_name.replace("-", "_"))
        name = name.strip("_")

        if not name:
            raise ValueError(
                "Could not sanitize collection name to be a valid Weaviate class name"
            )

        # Ensure it starts with a letter and is capitalized
        if not name[0].isalpha():
            name = "C" + name

        return name[0].upper() + name[1:]

    def has_collection(self, collection_name: str) -> bool:
        sane_collection_name = self._sanitize_collection_name(collection_name)
        return self._collections_exists(sane_collection_name)

    def delete_collection(self, collection_name: str) -> None:
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if self._collections_exists(sane_collection_name):
            self.client.collections.delete(sane_collection_name)

    def _create_collection(self, collection_name: str) -> None:
        # Use vectorizer_config=none so that objects accept a flat-list vector
        # via batch.add_object(vector=[...]).  The named-vector API
        # (Configure.Vectors.self_provided) would require dict-format vectors
        # and target_vector on every search — that's not how the rest of the
        # codebase works, so use the simple single-vector approach instead.
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            properties=[
                weaviate.classes.config.Property(
                    name="text", data_type=weaviate.classes.config.DataType.TEXT
                ),
                # Contextual chunking fields (Anthropic paper approach)
                weaviate.classes.config.Property(
                    name="raw_text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Original chunk text without context prefix — shown to users",
                ),
                weaviate.classes.config.Property(
                    name="context_summary",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="LLM-generated context sentence prepended before embedding",
                ),
                weaviate.classes.config.Property(
                    name="has_context",
                    data_type=weaviate.classes.config.DataType.BOOL,
                    description="True if contextual embedding was applied to this chunk",
                ),
                # ── Hierarchical chunking fields ──────────────────────────────
                weaviate.classes.config.Property(
                    name="parent_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="UUID of the immediate parent section (section_id)",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="parent_text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Full parent section text returned to the LLM for focused context",
                ),
                # ── Document structure / navigation labels ────────────────────
                weaviate.classes.config.Property(
                    name="header_1",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Top-level markdown heading (H1) of the section containing this chunk",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="header_2",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Second-level markdown heading (H2) of the section containing this chunk",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="breadcrumb",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Full heading path formatted as H1 > H2 > H3 for display in [Source:] headers",
                    skip_vectorization=True,
                ),
                # ── Layer 4: Multi-Granularity Indexing ───────────────────────
                # Stable chunk identity
                weaviate.classes.config.Property(
                    name="chunk_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Pre-assigned stable UUID — matches the Weaviate object UUID",
                    skip_vectorization=True,
                ),
                # Hierarchy IDs
                weaviate.classes.config.Property(
                    name="document_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="file_id of the source document",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="chapter_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Deterministic UUID5 for the H1/chapter section (stable across re-indexing)",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="section_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="UUID of the parent section block (alias of parent_id)",
                    skip_vectorization=True,
                ),
                # Relationship fields (comma-separated UUIDs)
                weaviate.classes.config.Property(
                    name="sibling_chunk_ids",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Comma-separated chunk_ids of sibling chunks (same parent, excluding self)",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="child_chunk_ids",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Comma-separated chunk_ids of all children under the same parent (including self)",
                    skip_vectorization=True,
                ),
                # Positional metadata
                weaviate.classes.config.Property(
                    name="position_in_section",
                    data_type=weaviate.classes.config.DataType.INT,
                    description="0-based index of this chunk within its parent section",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="token_count",
                    data_type=weaviate.classes.config.DataType.INT,
                    description="Approximate token count (chars // 4)",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="content_type",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Content classification: body_text | list_block | heading",
                    skip_vectorization=True,
                ),
                # Semantic metadata (populated after contextual chunking)
                weaviate.classes.config.Property(
                    name="topics",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="LLM-generated topic summary extracted from [CONTEXT] Topic: line",
                ),
                weaviate.classes.config.Property(
                    name="key_entities",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Named entities and key terms (alias of keywords field)",
                ),
                # ── Entity-anchored keyword field for Weaviate BM25 ───────────
                weaviate.classes.config.Property(
                    name="keywords",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Extracted named entities and key terms; searched by native BM25",
                ),
                # ── AKU Knowledge fields (AKU document collections + chunk linking) ──
                # Populated for AKU documents (AkuKnowledge collection):
                weaviate.classes.config.Property(
                    name="aku_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="AKU node ID (e.g. P6C1-TC1-A04); set for AKU knowledge docs",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="cluster_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Topic cluster ID (e.g. P6C1-TC1); set for AKU knowledge docs",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="source_chapter_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="NCERT chapter ID (e.g. CLASS_6-_POLITY_NCERT_C1)",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="structural_role",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="AKU structural role: foundational | integration | application",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="integration_node",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="'true' if this is the integration (cognitive hinge) node",
                    skip_vectorization=True,
                ),
                weaviate.classes.config.Property(
                    name="ncert_anchor",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Verbatim NCERT sentence grounding this AKU (gate_5_measurability.evidence)",
                ),
                weaviate.classes.config.Property(
                    name="wrong_belief",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Wrong mental model this AKU corrects; set for integration nodes",
                ),
                weaviate.classes.config.Property(
                    name="fitb_sentence",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Fill-in-the-blank sentence for integration nodes; contains '_______'",
                    skip_vectorization=True,
                ),
                # Populated retroactively by /api/v1/aku/link on book chunk collections:
                weaviate.classes.config.Property(
                    name="aku_ids",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Comma-separated AKU IDs this book chunk covers; set by aku_link stage",
                    skip_vectorization=True,
                ),
            ],
        )

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        import logging as _logging
        _log = _logging.getLogger(__name__)

        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            self._create_collection(sane_collection_name)

        collection = self.client.collections.get(sane_collection_name)

        with collection.batch.fixed_size(batch_size=100) as batch:
            for item in items:
                item_uuid = str(uuid.uuid4()) if not item["id"] else str(item["id"])

                properties = {"text": item["text"]}
                if item["metadata"]:
                    clean_metadata = _convert_uuids_to_strings(
                        process_metadata(item["metadata"])
                    )
                    clean_metadata.pop("text", None)
                    properties.update(clean_metadata)

                batch.add_object(
                    properties=properties, uuid=item_uuid, vector=item["vector"]
                )

        # Surface batch-level errors that Weaviate silently swallows by default
        failed = getattr(batch, "failed_objects", []) or []
        if failed:
            _log.error(
                "Weaviate batch insert: %d/%d objects FAILED for collection '%s'. "
                "First error: %s",
                len(failed), len(items), sane_collection_name,
                failed[0] if failed else "n/a",
            )
            print(
                f"  WEAVIATE INSERT ERROR: {len(failed)}/{len(items)} objects failed"
                f" in '{sane_collection_name}'. Check logs for details.",
                flush=True,
            )
        else:
            print(
                f"  WEAVIATE INSERT OK: {len(items)} objects → '{sane_collection_name}'",
                flush=True,
            )

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        import logging as _logging
        _log = _logging.getLogger(__name__)

        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            self._create_collection(sane_collection_name)

        collection = self.client.collections.get(sane_collection_name)

        with collection.batch.fixed_size(batch_size=100) as batch:
            for item in items:
                item_uuid = str(item["id"]) if item["id"] else None

                properties = {"text": item["text"]}
                if item["metadata"]:
                    clean_metadata = _convert_uuids_to_strings(
                        process_metadata(item["metadata"])
                    )
                    clean_metadata.pop("text", None)
                    properties.update(clean_metadata)

                batch.add_object(
                    properties=properties, uuid=item_uuid, vector=item["vector"]
                )

        failed = getattr(batch, "failed_objects", []) or []
        if failed:
            _log.error(
                "Weaviate batch upsert: %d/%d objects FAILED for collection '%s'. "
                "First error: %s",
                len(failed), len(items), sane_collection_name,
                failed[0] if failed else "n/a",
            )
            print(
                f"  WEAVIATE UPSERT ERROR: {len(failed)}/{len(items)} objects failed"
                f" in '{sane_collection_name}'.",
                flush=True,
            )

    def search(
        self,
        collection_name: str,
        vectors: List[List[Union[float, int]]],
        filter: Optional[dict] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        import logging as _logging
        _log = _logging.getLogger(__name__)

        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return None

        collection = self.client.collections.get(sane_collection_name)

        # Diagnostic: count objects so we can distinguish empty vs. vector-failure
        try:
            _agg = collection.aggregate.over_all(total_count=True)
            _obj_count = _agg.total_count if _agg else 0
            if _obj_count == 0:
                print(
                    f"  WEAVIATE EMPTY: '{sane_collection_name}' has 0 objects",
                    flush=True,
                )
                _log.warning("Collection '%s' has 0 objects — skipping near_vector", sane_collection_name)
                return SearchResult(ids=[[]], documents=[[]], metadatas=[[]], distances=[[]])
        except Exception as _cnt_exc:
            _obj_count = -1  # unknown
            _log.debug("Could not count objects in '%s': %s", sane_collection_name, _cnt_exc)

        result_ids, result_documents, result_metadatas, result_distances = (
            [],
            [],
            [],
            [],
        )

        for vector_embedding in vectors:
            try:
                response = collection.query.near_vector(
                    near_vector=vector_embedding,
                    filters=_build_weaviate_filter(filter),
                    limit=limit,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
                )

                ids = [str(obj.uuid) for obj in response.objects]
                documents = []
                metadatas = []
                distances = []

                for obj in response.objects:
                    properties = dict(obj.properties) if obj.properties else {}
                    documents.append(properties.pop("text", ""))
                    metadatas.append(_convert_uuids_to_strings(properties))

                # Weaviate has cosine distance, 2 (worst) -> 0 (best). Re-ordering to 0 -> 1
                raw_distances = [
                    (
                        obj.metadata.distance
                        if obj.metadata and obj.metadata.distance
                        else 2.0
                    )
                    for obj in response.objects
                ]
                distances = [(2 - dist) / 2 for dist in raw_distances]

                result_ids.append(ids)
                result_documents.append(documents)
                result_metadatas.append(metadatas)
                result_distances.append(distances)
            except Exception as _search_exc:
                _log.error(
                    "near_vector search FAILED for collection '%s' "
                    "(vec_dim=%d): %s",
                    sane_collection_name,
                    len(vector_embedding) if vector_embedding else 0,
                    _search_exc,
                    exc_info=True,
                )
                print(
                    f"  WEAVIATE SEARCH ERROR in '{sane_collection_name}': {_search_exc}",
                    flush=True,
                )
                result_ids.append([])
                result_documents.append([])
                result_metadatas.append([])
                result_distances.append([])

        return SearchResult(
            **{
                "ids": result_ids,
                "documents": result_documents,
                "metadatas": result_metadatas,
                "distances": result_distances,
            }
        )

    def query(
        self, collection_name: str, filter: Dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return None

        collection = self.client.collections.get(sane_collection_name)

        try:
            response = collection.query.fetch_objects(
                filters=_build_weaviate_filter(filter), limit=limit
            )

            ids = [str(obj.uuid) for obj in response.objects]
            documents = []
            metadatas = []

            for obj in response.objects:
                properties = dict(obj.properties) if obj.properties else {}
                documents.append(properties.pop("text", ""))
                metadatas.append(_convert_uuids_to_strings(properties))

            return GetResult(
                **{
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                }
            )
        except Exception as _qexc:
            _log.warning(
                "Weaviate query() failed for collection '%s': %s",
                sane_collection_name, _qexc,
            )
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return None

        collection = self.client.collections.get(sane_collection_name)
        ids, documents, metadatas = [], [], []

        try:
            for item in collection.iterator():
                ids.append(str(item.uuid))
                properties = dict(item.properties) if item.properties else {}
                documents.append(properties.pop("text", ""))
                metadatas.append(_convert_uuids_to_strings(properties))

            if not ids:
                return None

            return GetResult(
                **{
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                }
            )
        except Exception:
            return None

    def peek(self, collection_name: str, limit: int = 10) -> Optional[GetResult]:
        """Fetch a small sample — O(1) vs O(n) for get(). Used for null-checks and
        detecting hierarchical collections without loading all chunks into memory."""
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return None
        collection = self.client.collections.get(sane_collection_name)
        try:
            response = collection.query.fetch_objects(limit=limit)
            if not response.objects:
                return None
            ids, documents, metadatas = [], [], []
            for obj in response.objects:
                props = dict(obj.properties) if obj.properties else {}
                ids.append(str(obj.uuid))
                documents.append(props.pop("text", ""))
                metadatas.append(_convert_uuids_to_strings(props))
            return GetResult(ids=[ids], documents=[documents], metadatas=[metadatas])
        except Exception:
            return None

    def bm25_search(
        self,
        collection_name: str,
        query: str,
        query_properties: List[str],
        limit: int = 15,
        filter: Optional[dict] = None,
    ) -> Optional[SearchResult]:
        """Native Weaviate BM25 search on the pre-built inverted index.
        Searches query_properties (e.g. ['raw_text', 'keywords']); returns
        raw_text as the document content so [Context: ...] prefixes never
        appear in BM25 scoring."""
        import logging as _logging
        _log = _logging.getLogger(__name__)
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return None
        collection = self.client.collections.get(sane_collection_name)
        try:
            response = collection.query.bm25(
                query=query,
                query_properties=query_properties,
                filters=_build_weaviate_filter(filter),
                limit=limit,
                return_metadata=weaviate.classes.query.MetadataQuery(score=True),
            )
            ids, documents, metadatas, distances = [], [], [], []
            for obj in response.objects:
                props = dict(obj.properties) if obj.properties else {}
                ids.append(str(obj.uuid))
                # Use raw_text for display; never expose the [Context: ...] prefix
                display = props.get("raw_text") or props.pop("text", "")
                props.pop("text", None)
                documents.append(display)
                metadatas.append(_convert_uuids_to_strings(props))
                score = float(obj.metadata.score) if obj.metadata and obj.metadata.score else 0.0
                distances.append(score)
            if not ids:
                return None
            return SearchResult(
                ids=[ids], documents=[documents], metadatas=[metadatas], distances=[distances]
            )
        except Exception as e:
            _log.warning("Weaviate BM25 search failed for '%s': %s", sane_collection_name, e)
            return None

    def update_object_properties(
        self,
        collection_name: str,
        object_id: str,
        properties: dict,
    ) -> bool:
        """Patch specific properties of an existing Weaviate object WITHOUT re-embedding.

        Uses Weaviate's partial update (data.update) — the stored vector is
        unchanged; only the named scalar properties are overwritten.  Safe to
        call repeatedly: subsequent calls simply overwrite the same fields.

        Returns True on success, False on any error (never raises).
        """
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            _log.warning(
                "update_object_properties: collection '%s' does not exist",
                sane_collection_name,
            )
            return False
        collection = self.client.collections.get(sane_collection_name)
        try:
            collection.data.update(uuid=object_id, properties=properties)
            return True
        except Exception as exc:
            _log.warning(
                "update_object_properties failed  collection=%s  uuid=%s  error=%s",
                sane_collection_name, object_id, exc,
            )
            return False

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
    ) -> None:
        sane_collection_name = self._sanitize_collection_name(collection_name)
        if not self._collections_exists(sane_collection_name):
            return

        collection = self.client.collections.get(sane_collection_name)

        try:
            if ids:
                for item_id in ids:
                    collection.data.delete_by_id(uuid=item_id)
            elif filter:
                weaviate_filter = _build_weaviate_filter(filter)
                if weaviate_filter:
                    collection.data.delete_many(where=weaviate_filter)
        except Exception:
            pass

    def reset(self) -> None:
        try:
            for collection_name in self.client.collections.list_all().keys():
                self.client.collections.delete(collection_name)
        except Exception:
            pass
