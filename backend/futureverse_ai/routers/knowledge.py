from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
import logging
import os
import io
import re
import zipfile

from sqlalchemy.orm import Session
from futureverse_ai.internal.db import get_session
from futureverse_ai.models.groups import Groups
from futureverse_ai.models.knowledge import (
    KnowledgeFileListResponse,
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse,
)
from futureverse_ai.models.files import Files, FileModel, FileMetadataResponse
from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT
from futureverse_ai.routers.retrieval import (
    process_file,
    ProcessFileForm,
    process_files_batch,
    BatchProcessFilesForm,
)
from futureverse_ai.storage.provider import Storage

from futureverse_ai.constants import ERROR_MESSAGES
from futureverse_ai.utils.auth import get_verified_user, get_admin_user
from futureverse_ai.utils.access_control import has_permission
from futureverse_ai.models.access_grants import AccessGrants, has_public_read_access_grant


from futureverse_ai.config import BYPASS_ADMIN_ACCESS_CONTROL
from futureverse_ai.models.models import Models, ModelForm

log = logging.getLogger(__name__)

router = APIRouter()

############################
# getKnowledgeBases
############################

PAGE_ITEM_COUNT = 30

############################
# Knowledge Base Embedding
############################

KNOWLEDGE_BASES_COLLECTION = "knowledge-bases"


async def embed_knowledge_base_metadata(
    request: Request,
    knowledge_base_id: str,
    name: str,
    description: str,
) -> bool:
    """Generate and store embedding for knowledge base."""
    try:
        content = f"{name}\n\n{description}" if description else name
        embedding = await request.app.state.EMBEDDING_FUNCTION(content)
        VECTOR_DB_CLIENT.upsert(
            collection_name=KNOWLEDGE_BASES_COLLECTION,
            items=[
                {
                    "id": knowledge_base_id,
                    "text": content,
                    "vector": embedding,
                    "metadata": {
                        "knowledge_base_id": knowledge_base_id,
                    },
                }
            ],
        )
        return True
    except Exception as e:
        log.error(f"Failed to embed knowledge base {knowledge_base_id}: {e}")
        return False


def remove_knowledge_base_metadata_embedding(knowledge_base_id: str) -> bool:
    """Remove knowledge base embedding."""
    try:
        VECTOR_DB_CLIENT.delete(
            collection_name=KNOWLEDGE_BASES_COLLECTION,
            ids=[knowledge_base_id],
        )
        return True
    except Exception as e:
        log.debug(f"Failed to remove embedding for {knowledge_base_id}: {e}")
        return False


def _purge_rag_result_cache(kb_id: str, file_id: Optional[str] = None) -> None:
    """Invalidate cached RAG retrieval results for a knowledge base after it
    is mutated, so stale answers are never served from the result cache.

    No-op unless ENABLE_RAG_RESULT_CACHE is on; fail-open on any Redis error.
    Purges the KB's own collection plus the affected file's collection under
    both naming schemes ("file-{id}" raw and the sanitized "File_..." form
    used by the Weaviate client)."""
    try:
        from futureverse_ai.retrieval.rag_cache import purge_collections

        collections = [kb_id]
        if file_id:
            raw = f"file-{file_id}"
            collections.append(raw)
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "", raw.replace("-", "_")).strip("_")
            if sanitized:
                if not sanitized[0].isalpha():
                    sanitized = "C" + sanitized
                else:
                    sanitized = sanitized[0].upper() + sanitized[1:]
                collections.append(sanitized)
        purge_collections(collections)
    except Exception as e:
        log.debug(f"RAG result cache purge failed (fail-open): {e}")


class KnowledgeAccessResponse(KnowledgeUserResponse):
    write_access: Optional[bool] = False


class KnowledgeAccessListResponse(BaseModel):
    items: list[KnowledgeAccessResponse]
    total: int


@router.get("/", response_model=KnowledgeAccessListResponse)
async def get_knowledge_bases(
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    groups = Groups.get_groups_by_member_id(user.id, db=db)
    user_group_ids = {group.id for group in groups}

    if not user.role == "admin" or not BYPASS_ADMIN_ACCESS_CONTROL:
        if groups:
            filter["group_ids"] = [group.id for group in groups]

        filter["user_id"] = user.id

    result = Knowledges.search_knowledge_bases(
        user.id, filter=filter, skip=skip, limit=limit, db=db
    )

    # Batch-fetch writable knowledge IDs in a single query instead of N has_access calls
    knowledge_base_ids = [knowledge_base.id for knowledge_base in result.items]
    writable_knowledge_base_ids = AccessGrants.get_accessible_resource_ids(
        user_id=user.id,
        resource_type="knowledge",
        resource_ids=knowledge_base_ids,
        permission="write",
        user_group_ids=user_group_ids,
        db=db,
    )

    return KnowledgeAccessListResponse(
        items=[
            KnowledgeAccessResponse(
                **knowledge_base.model_dump(),
                write_access=(
                    user.id == knowledge_base.user_id
                    or (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
                    or knowledge_base.id in writable_knowledge_base_ids
                ),
            )
            for knowledge_base in result.items
        ],
        total=result.total,
    )


@router.get("/search", response_model=KnowledgeAccessListResponse)
async def search_knowledge_bases(
    query: Optional[str] = None,
    view_option: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query
    if view_option:
        filter["view_option"] = view_option

    groups = Groups.get_groups_by_member_id(user.id, db=db)
    user_group_ids = {group.id for group in groups}

    if not user.role == "admin" or not BYPASS_ADMIN_ACCESS_CONTROL:
        if groups:
            filter["group_ids"] = [group.id for group in groups]

        filter["user_id"] = user.id

    result = Knowledges.search_knowledge_bases(
        user.id, filter=filter, skip=skip, limit=limit, db=db
    )

    # Batch-fetch writable knowledge IDs in a single query instead of N has_access calls
    knowledge_base_ids = [knowledge_base.id for knowledge_base in result.items]
    writable_knowledge_base_ids = AccessGrants.get_accessible_resource_ids(
        user_id=user.id,
        resource_type="knowledge",
        resource_ids=knowledge_base_ids,
        permission="write",
        user_group_ids=user_group_ids,
        db=db,
    )

    return KnowledgeAccessListResponse(
        items=[
            KnowledgeAccessResponse(
                **knowledge_base.model_dump(),
                write_access=(
                    user.id == knowledge_base.user_id
                    or (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
                    or knowledge_base.id in writable_knowledge_base_ids
                ),
            )
            for knowledge_base in result.items
        ],
        total=result.total,
    )


@router.get("/search/files", response_model=KnowledgeFileListResponse)
async def search_knowledge_files(
    query: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query

    groups = Groups.get_groups_by_member_id(user.id, db=db)
    if groups:
        filter["group_ids"] = [group.id for group in groups]

    filter["user_id"] = user.id

    return Knowledges.search_knowledge_files(
        filter=filter, skip=skip, limit=limit, db=db
    )


############################
# CreateNewKnowledge
############################


@router.post("/create", response_model=Optional[KnowledgeResponse])
async def create_new_knowledge(
    request: Request,
    form_data: KnowledgeForm,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_session) here.
    # Database operations (has_permission, insert_new_knowledge) manage their own sessions.
    # This prevents holding a connection during embed_knowledge_base_metadata()
    # which makes external embedding API calls (1-5+ seconds).
    if user.role != "admin" and not has_permission(
        user.id, "workspace.knowledge", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    # Check if user can share publicly
    if (
        user.role != "admin"
        and has_public_read_access_grant(form_data.access_grants)
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        form_data.access_grants = []

    knowledge = Knowledges.insert_new_knowledge(user.id, form_data)

    if knowledge:
        # Embed knowledge base for semantic search
        await embed_knowledge_base_metadata(
            request,
            knowledge.id,
            knowledge.name,
            knowledge.description,
        )
        return knowledge
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_EXISTS,
        )


############################
# ReindexKnowledgeFiles
############################


@router.post("/reindex", response_model=bool)
async def reindex_knowledge_files(
    request: Request,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    knowledge_bases = Knowledges.get_knowledge_bases(db=db)

    log.info(f"Starting reindexing for {len(knowledge_bases)} knowledge bases")

    for knowledge_base in knowledge_bases:
        try:
            files = Knowledges.get_files_by_id(knowledge_base.id, db=db)
            try:
                if VECTOR_DB_CLIENT.has_collection(collection_name=knowledge_base.id):
                    VECTOR_DB_CLIENT.delete_collection(
                        collection_name=knowledge_base.id
                    )
            except Exception as e:
                log.error(f"Error deleting collection {knowledge_base.id}: {str(e)}")
                continue  # Skip, don't raise

            failed_files = []
            for file in files:
                try:
                    await run_in_threadpool(
                        process_file,
                        request,
                        ProcessFileForm(
                            file_id=file.id, collection_name=knowledge_base.id
                        ),
                        user=user,
                        db=db,
                    )
                except Exception as e:
                    log.error(
                        f"Error processing file {file.filename} (ID: {file.id}): {str(e)}"
                    )
                    failed_files.append({"file_id": file.id, "error": str(e)})
                    continue

            _purge_rag_result_cache(knowledge_base.id)

        except Exception as e:
            log.error(f"Error processing knowledge base {knowledge_base.id}: {str(e)}")
            # Don't raise, just continue
            continue

        if failed_files:
            log.warning(
                f"Failed to process {len(failed_files)} files in knowledge base {knowledge_base.id}"
            )
            for failed in failed_files:
                log.warning(f"File ID: {failed['file_id']}, Error: {failed['error']}")

    log.info(f"Reindexing completed.")
    return True


############################
# ReindexKnowledgeBases
############################


@router.post("/metadata/reindex", response_model=dict)
async def reindex_knowledge_base_metadata_embeddings(
    request: Request,
    user=Depends(get_admin_user),
):
    """Batch embed all existing knowledge bases. Admin only.

    NOTE: We intentionally do NOT use Depends(get_session) here.
    This endpoint loops through ALL knowledge bases and calls embed_knowledge_base_metadata()
    for each one, making N external embedding API calls. Holding a session during
    this entire operation would exhaust the connection pool.
    """
    knowledge_bases = Knowledges.get_knowledge_bases()
    log.info(f"Reindexing embeddings for {len(knowledge_bases)} knowledge bases")

    success_count = 0
    for kb in knowledge_bases:
        if await embed_knowledge_base_metadata(request, kb.id, kb.name, kb.description):
            success_count += 1
            _purge_rag_result_cache(kb.id)

    log.info(f"Embedding reindex complete: {success_count}/{len(knowledge_bases)}")
    return {"total": len(knowledge_bases), "success": success_count}


############################
# GetKnowledgeById
############################


class KnowledgeFilesResponse(KnowledgeResponse):
    files: Optional[list[FileMetadataResponse]] = None
    write_access: Optional[bool] = False


@router.get("/{id}", response_model=Optional[KnowledgeFilesResponse])
async def get_knowledge_by_id(
    id: str, user=Depends(get_verified_user), db: Session = Depends(get_session)
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)

    if knowledge:
        if (
            user.role == "admin"
            or knowledge.user_id == user.id
            or AccessGrants.has_access(
                user_id=user.id,
                resource_type="knowledge",
                resource_id=knowledge.id,
                permission="read",
                db=db,
            )
        ):

            return KnowledgeFilesResponse(
                **knowledge.model_dump(),
                write_access=(
                    user.id == knowledge.user_id
                    or (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
                    or AccessGrants.has_access(
                        user_id=user.id,
                        resource_type="knowledge",
                        resource_id=knowledge.id,
                        permission="write",
                        db=db,
                    )
                ),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateKnowledgeById
############################


@router.post("/{id}/update", response_model=Optional[KnowledgeFilesResponse])
async def update_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeForm,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_session) here.
    # Database operations manage their own short-lived sessions internally.
    # This prevents holding a connection during embed_knowledge_base_metadata()
    # which makes external embedding API calls (1-5+ seconds).
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    # Is the user the original creator, in a group with write access, or an admin
    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Check if user can share publicly
    if (
        user.role != "admin"
        and has_public_read_access_grant(form_data.access_grants)
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        form_data.access_grants = []

    knowledge = Knowledges.update_knowledge_by_id(id=id, form_data=form_data)
    if knowledge:
        # Re-embed knowledge base for semantic search
        await embed_knowledge_base_metadata(
            request,
            knowledge.id,
            knowledge.name,
            knowledge.description,
        )
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )


############################
# UpdateKnowledgeAccessById
############################


class KnowledgeAccessGrantsForm(BaseModel):
    access_grants: list[dict]


@router.post("/{id}/access/update", response_model=Optional[KnowledgeFilesResponse])
async def update_knowledge_access_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeAccessGrantsForm,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Strip public sharing if user lacks permission
    if (
        user.role != "admin"
        and has_public_read_access_grant(form_data.access_grants)
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        form_data.access_grants = [
            grant
            for grant in form_data.access_grants
            if not (
                grant.get("principal_type") == "user"
                and grant.get("principal_id") == "*"
            )
        ]

    AccessGrants.set_access_grants("knowledge", id, form_data.access_grants, db=db)

    return KnowledgeFilesResponse(
        **Knowledges.get_knowledge_by_id(id=id, db=db).model_dump(),
        files=Knowledges.get_file_metadatas_by_id(id, db=db),
    )


############################
# GetKnowledgeFilesById
############################


@router.get("/{id}/files", response_model=KnowledgeFileListResponse)
async def get_knowledge_files_by_id(
    id: str,
    query: Optional[str] = None,
    view_option: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):

    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if not (
        user.role == "admin"
        or knowledge.user_id == user.id
        or AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="read",
            db=db,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    page = max(page, 1)

    limit = 30
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query
    if view_option:
        filter["view_option"] = view_option
    if order_by:
        filter["order_by"] = order_by
    if direction:
        filter["direction"] = direction

    return Knowledges.search_files_by_id(
        id, user.id, filter=filter, skip=skip, limit=limit, db=db
    )


############################
# AddFileToKnowledge
############################


class KnowledgeFileIdForm(BaseModel):
    file_id: str


@router.post("/{id}/file/add", response_model=Optional[KnowledgeFilesResponse])
def add_file_to_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id, db=db)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    if not file.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_PROCESSED,
        )

    # Add content to the vector database
    try:
        process_file(
            request,
            ProcessFileForm(file_id=form_data.file_id, collection_name=id),
            user=user,
            db=db,
        )

        # Add file to knowledge base
        Knowledges.add_file_to_knowledge_by_id(
            knowledge_id=id, file_id=form_data.file_id, user_id=user.id, db=db
        )

        _purge_rag_result_cache(id, form_data.file_id)

        # === AUTO NEO4J GRAPH INGESTION ===
        import threading as _threading
        def _ingest_to_neo4j(_file_id: str, _doc_name: str, _collection_name: str):
            import os, logging, traceback
            logger = logging.getLogger("neo4j_ingest")
            try:
                logger.info(f"[Neo4j] Thread started for file {_file_id}")
                if not os.environ.get("NEO4J_URI"):
                    logger.warning("[Neo4j] NEO4J_URI not set. Skipping.")
                    return
                import time
                import weaviate
                from futureverse_ai.retrieval.graph.extractor import extract_and_store

                client = weaviate.connect_to_local(
                    host=os.environ.get("WEAVIATE_HTTP_HOST", "localhost"),
                    port=int(os.environ.get("WEAVIATE_HTTP_PORT", "8085")),
                    grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
                )

                collection_name = f"File_{_file_id.replace('-', '_')}"

                # Wait for the embedding pipeline to finish writing to Weaviate by
                # POLLING the collection's object count until it stops growing,
                # instead of a blind fixed sleep (was time.sleep(30) — a race).
                # Proceeds after NEO4J_INGEST_EMBED_WAIT seconds regardless.
                _max_wait = int(os.environ.get("NEO4J_INGEST_EMBED_WAIT", "120"))
                _poll = max(1, int(os.environ.get("NEO4J_INGEST_POLL_INTERVAL", "3")))
                _deadline = time.time() + _max_wait
                _prev, _stable = -1, 0
                while time.time() < _deadline:
                    try:
                        if not client.collections.exists(collection_name):
                            time.sleep(_poll)
                            continue
                        _cnt = (
                            client.collections.get(collection_name)
                            .aggregate.over_all(total_count=True)
                            .total_count
                        )
                        if _cnt and _cnt == _prev:
                            _stable += 1
                            if _stable >= 2:        # unchanged for 2 polls -> embedding done
                                break
                        else:
                            _stable = 0
                        _prev = _cnt
                    except Exception as _pe:
                        logger.debug(f"[Neo4j] embedding poll: {_pe}")
                    time.sleep(_poll)
                logger.info(f"[Neo4j] Embedding ready (~{_prev} objects) for {collection_name}")

                logger.info(f"[Neo4j] Reading chunks from {collection_name}")
                col = client.collections.get(collection_name)
                # Fetch ALL chunks via iterator (no hard 500-object cap)
                all_objects = list(col.iterator())
                items_objects = all_objects

                chunks = []
                for obj in items_objects:
                    props = obj.properties or {}
                    # Prefer raw_text (original chunk without context prefix) for KG
                    # extraction — the context prefix is useful for embedding/retrieval
                    # but adds noise to entity/relation extraction.
                    text = props.get("raw_text") or props.get("text", "")
                    if not text:
                        continue
                    chunks.append({
                        "text":        text,
                        "subject":     props.get("subject", ""),
                        "book":        props.get("book", ""),
                        "chapter":     props.get("chapter", ""),
                        "class_level": str(props.get("class_level", "")),
                        "page_number": int(props.get("page_number", 0)),
                        "page_id":     props.get("page_id", ""),
                    })
                client.close()

                def _infer_subject(doc_name: str) -> str:
                    dn = doc_name.lower()
                    if any(x in dn for x in ["geography", "earth", "resource", "agriculture", "water", "mineral", "forest", "climate", "drainage"]):
                        return "Geography"
                    elif any(x in dn for x in ["history", "medieval", "ancient", "modern", "mughal", "colonial"]):
                        return "History"
                    elif any(x in dn for x in ["polity", "constitution", "governance", "parliament"]):
                        return "Polity"
                    elif any(x in dn for x in ["economy", "economic", "development"]):
                        return "Economy"
                    elif any(x in dn for x in ["science", "physics", "chemistry", "biology", "technology"]):
                        return "Science"
                    elif any(x in dn for x in ["ecology", "biodiversity", "pollution", "ecosystem"]):
                        return "Environment"
                    return "General"

                doc_subject = _infer_subject(_doc_name)
                logger.info(f"[Neo4j] {len(chunks)} chunks | Subject: {doc_subject} | Doc: {_doc_name}")

                if not chunks:
                    logger.warning("[Neo4j] No chunks found")
                    return

                def is_good_chunk(text):
                    if not text or len(text.strip()) < 120:
                        return False
                    words = text.split()
                    if len(words) < 25:
                        return False
                    alpha_words = [w for w in words if w.isalpha() and len(w) > 2]
                    if len(alpha_words) / max(len(words), 1) < 0.45:
                        return False
                    return True
                valid_chunks = [(i, c) for i, c in enumerate(chunks) if is_good_chunk(c.get("text", ""))]
                logger.info(f"[Neo4j] Processing {len(valid_chunks)} valid chunks")

                success = failed = 0

                def infer_from_chapter(chapter_name):
                    ch = (chapter_name or "").lower()
                    if any(x in ch for x in ["resource", "agriculture", "water", "forest", "wildlife",
                        "mineral", "manufacturing", "life lines", "drainage", "climate", "vegetation",
                        "land", "soil", "ocean", "atmosphere", "physical", "distribution", "migration",
                        "population", "settlement", "transport", "trade", "map", "globe", "geography",
                        "interior", "rocks", "earthquake", "latitude", "longitude", "motions"]):
                        return "Geography"
                    elif any(x in ch for x in ["french", "russia", "nazism", "war", "imperialism",
                        "nationalism", "mughal", "colonial", "ancient", "medieval", "revolt",
                        "dynasty", "empire", "civilization", "history", "britain", "partition"]):
                        return "History"
                    elif any(x in ch for x in ["constitution", "parliament", "federalism", "democracy",
                        "rights", "judiciary", "election", "political", "governance", "power sharing",
                        "gender", "caste", "religion", "polity", "local government"]):
                        return "Polity"
                    elif any(x in ch for x in ["development", "sectors", "money", "credit",
                        "globalisation", "consumer", "poverty", "food security", "employment",
                        "economy", "market", "income", "infrastructure"]):
                        return "Economy"
                    elif any(x in ch for x in ["biodiversity", "ecosystem", "pollution", "ozone",
                        "sustainable", "ecology", "environment", "biosphere", "natural hazard"]):
                        return "Environment"
                    return None

                def process_chunk(args):
                    idx, chunk = args
                    chunk_chapter = chunk.get("chapter", "") or ""
                    chunk_subject = chunk.get("subject", "") or ""
                    # Chapter name is most reliable subject signal
                    inferred = infer_from_chapter(chunk_chapter)
                    final_subject = inferred or chunk_subject or doc_subject
                    return extract_and_store(
                        text=chunk["text"],
                        doc_id=f"{_file_id}_{idx}",
                        doc_name=_doc_name,
                        subject=final_subject,
                        book=chunk.get("book", "") or _doc_name,
                        chapter=chunk_chapter,
                        class_level=chunk.get("class_level", "") or "",
                        page_number=chunk["page_number"] or idx,
                    )

                # Process sequentially — Atlas-RAG calls Ollama (GPU) and running
                # 2 parallel extractions would compete with the embedding model for VRAM.
                for i, item in enumerate(valid_chunks):
                    try:
                        result = process_chunk(item)
                        if result.get("status") == "success":
                            success += 1
                        else:
                            failed += 1
                    except Exception as chunk_err:
                        failed += 1
                        logger.warning(f"[Neo4j] Chunk error: {chunk_err}")
                    if i % 10 == 0:
                        logger.info(f"[Neo4j] Progress: {i+1}/{len(valid_chunks)} | ok={success} fail={failed}")

                logger.info(f"[Neo4j] COMPLETE: {success}/{len(valid_chunks)} for {_doc_name} [{doc_subject}]")

            except Exception as e:
                logger.error("[Neo4j] THREAD CRASHED")
                logger.error(str(e))
                logger.error(traceback.format_exc())
        if getattr(request.app.state.config, "ENABLE_NEO4J", False):
            log.info(f"Neo4j: starting background thread for file {form_data.file_id}")
            _t = _threading.Thread(
                target=_ingest_to_neo4j,
                args=(form_data.file_id, file.filename if file else "", id),
                daemon=True
            )
            _t.start()
        else:
            log.info("Neo4j KG ingestion skipped (disabled in admin settings)")
        # === END AUTO NEO4J GRAPH INGESTION ===

    except Exception as e:
        log.debug(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id, db=db),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.post("/{id}/file/update", response_model=Optional[KnowledgeFilesResponse])
def update_file_from_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id, db=db)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Remove content from the vector database
    VECTOR_DB_CLIENT.delete(
        collection_name=knowledge.id, filter={"file_id": form_data.file_id}
    )

    # Add content to the vector database
    try:
        process_file(
            request,
            ProcessFileForm(file_id=form_data.file_id, collection_name=id),
            user=user,
            db=db,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    _purge_rag_result_cache(id, form_data.file_id)

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id, db=db),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# RemoveFileFromKnowledge
############################


@router.post("/{id}/file/remove", response_model=Optional[KnowledgeFilesResponse])
def remove_file_from_knowledge_by_id(
    id: str,
    form_data: KnowledgeFileIdForm,
    delete_file: bool = Query(True),
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id, db=db)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    Knowledges.remove_file_from_knowledge_by_id(
        knowledge_id=id, file_id=form_data.file_id, db=db
    )

    _purge_rag_result_cache(id, form_data.file_id)

    # Remove content from the vector database
    try:
        VECTOR_DB_CLIENT.delete(
            collection_name=knowledge.id, filter={"file_id": form_data.file_id}
        )  # Remove by file_id first

        VECTOR_DB_CLIENT.delete(
            collection_name=knowledge.id, filter={"hash": file.hash}
        )  # Remove by hash as well in case of duplicates
    except Exception as e:
        log.debug("This was most likely caused by bypassing embedding processing")
        log.debug(e)
        pass

    if delete_file:
        try:
            # Remove the file's collection from vector database
            file_collection = f"file-{form_data.file_id}"
            if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection):
                VECTOR_DB_CLIENT.delete_collection(collection_name=file_collection)
        except Exception as e:
            log.debug("This was most likely caused by bypassing embedding processing")
            log.debug(e)
            pass

        # Delete file from database
        Files.delete_file_by_id(form_data.file_id, db=db)

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id, db=db),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# DeleteKnowledgeById
############################


@router.delete("/{id}/delete", response_model=bool)
async def delete_knowledge_by_id(
    id: str, user=Depends(get_verified_user), db: Session = Depends(get_session)
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    log.info(f"Deleting knowledge base: {id} (name: {knowledge.name})")

    # Get all models
    models = Models.get_all_models(db=db)
    log.info(f"Found {len(models)} models to check for knowledge base {id}")

    # Update models that reference this knowledge base
    for model in models:
        if model.meta and hasattr(model.meta, "knowledge"):
            knowledge_list = model.meta.knowledge or []
            # Filter out the deleted knowledge base
            updated_knowledge = [k for k in knowledge_list if k.get("id") != id]

            # If the knowledge list changed, update the model
            if len(updated_knowledge) != len(knowledge_list):
                log.info(f"Updating model {model.id} to remove knowledge base {id}")
                model.meta.knowledge = updated_knowledge
                # Create a ModelForm for the update
                model_form = ModelForm(
                    id=model.id,
                    name=model.name,
                    base_model_id=model.base_model_id,
                    meta=model.meta,
                    params=model.params,
                    access_grants=model.access_grants,
                    is_active=model.is_active,
                )
                Models.update_model_by_id(model.id, model_form, db=db)

    # Clean up vector DB
    try:
        VECTOR_DB_CLIENT.delete_collection(collection_name=id)
    except Exception as e:
        log.debug(e)
        pass

    # Remove knowledge base embedding
    remove_knowledge_base_metadata_embedding(id)

    result = Knowledges.delete_knowledge_by_id(id=id, db=db)
    return result


############################
# ResetKnowledgeById
############################


@router.post("/{id}/reset", response_model=Optional[KnowledgeResponse])
async def reset_knowledge_by_id(
    id: str, user=Depends(get_verified_user), db: Session = Depends(get_session)
):
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        VECTOR_DB_CLIENT.delete_collection(collection_name=id)
    except Exception as e:
        log.debug(e)
        pass

    knowledge = Knowledges.reset_knowledge_by_id(id=id, db=db)
    return knowledge


############################
# AddFilesToKnowledge
############################


@router.post("/{id}/files/batch/add", response_model=Optional[KnowledgeFilesResponse])
async def add_files_to_knowledge_batch(
    request: Request,
    id: str,
    form_data: list[KnowledgeFileIdForm],
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    """
    Add multiple files to a knowledge base
    """
    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge.id,
            permission="write",
            db=db,
        )
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Batch-fetch all files to avoid N+1 queries
    log.info(f"files/batch/add - {len(form_data)} files")
    file_ids = [form.file_id for form in form_data]
    files = Files.get_files_by_ids(file_ids, db=db)

    # Verify all requested files were found
    found_ids = {file.id for file in files}
    missing_ids = [fid for fid in file_ids if fid not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File {missing_ids[0]} not found",
        )

    # Process files
    try:
        result = await process_files_batch(
            request=request,
            form_data=BatchProcessFilesForm(files=files, collection_name=id),
            user=user,
            db=db,
        )
    except Exception as e:
        log.error(
            f"add_files_to_knowledge_batch: Exception occurred: {e}", exc_info=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Only add files that were successfully processed
    successful_file_ids = [r.file_id for r in result.results if r.status == "completed"]
    for file_id in successful_file_ids:
        Knowledges.add_file_to_knowledge_by_id(
            knowledge_id=id, file_id=file_id, user_id=user.id, db=db
        )

    # If there were any errors, include them in the response
    if result.errors:
        error_details = [f"{err.file_id}: {err.error}" for err in result.errors]
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id, db=db),
            warnings={
                "message": "Some files failed to process",
                "errors": error_details,
            },
        )

    return KnowledgeFilesResponse(
        **knowledge.model_dump(),
        files=Knowledges.get_file_metadatas_by_id(knowledge.id, db=db),
    )


############################
# ExportKnowledgeById
############################


@router.get("/{id}/export")
async def export_knowledge_by_id(
    id: str, user=Depends(get_admin_user), db: Session = Depends(get_session)
):
    """
    Export a knowledge base as a zip file containing .txt files.
    Admin only.
    """

    knowledge = Knowledges.get_knowledge_by_id(id=id, db=db)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    files = Knowledges.get_files_by_id(id, db=db)

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            content = file.data.get("content", "") if file.data else ""
            if content:
                # Use original filename with .txt extension
                filename = file.filename
                if not filename.endswith(".txt"):
                    filename = f"{filename}.txt"
                zf.writestr(filename, content)

    zip_buffer.seek(0)

    # Sanitize knowledge name for filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in knowledge.name)
    zip_filename = f"{safe_name}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
    )
