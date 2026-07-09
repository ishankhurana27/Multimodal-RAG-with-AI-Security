import json
import logging
import mimetypes
import os
import shutil
import asyncio
import time as _time

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

from fastapi import (
    Depends,
    FastAPI,
    Query,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import tiktoken


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

from futureverse_ai.models.files import FileModel, FileUpdateForm, Files
from futureverse_ai.models.knowledge import Knowledges
from futureverse_ai.storage.provider import Storage
from futureverse_ai.internal.db import get_session, get_db
from sqlalchemy.orm import Session


from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT

# Document loaders
from futureverse_ai.retrieval.loaders.main import Loader
from futureverse_ai.retrieval.loaders.youtube import YoutubeLoader

# Web search engines
from futureverse_ai.retrieval.web.main import SearchResult
from futureverse_ai.retrieval.web.utils import get_web_loader
from futureverse_ai.retrieval.web.ollama import search_ollama_cloud
from futureverse_ai.retrieval.web.perplexity_search import search_perplexity_search
from futureverse_ai.retrieval.web.brave import search_brave
from futureverse_ai.retrieval.web.kagi import search_kagi
from futureverse_ai.retrieval.web.mojeek import search_mojeek
from futureverse_ai.retrieval.web.bocha import search_bocha
from futureverse_ai.retrieval.web.duckduckgo import search_duckduckgo
from futureverse_ai.retrieval.web.google_pse import search_google_pse
from futureverse_ai.retrieval.web.jina_search import search_jina
from futureverse_ai.retrieval.web.searchapi import search_searchapi
from futureverse_ai.retrieval.web.serpapi import search_serpapi
from futureverse_ai.retrieval.web.searxng import search_searxng
from futureverse_ai.retrieval.web.yacy import search_yacy
from futureverse_ai.retrieval.web.serper import search_serper
from futureverse_ai.retrieval.web.serply import search_serply
from futureverse_ai.retrieval.web.serpstack import search_serpstack
from futureverse_ai.retrieval.web.tavily import search_tavily
from futureverse_ai.retrieval.web.bing import search_bing
from futureverse_ai.retrieval.web.azure import search_azure
from futureverse_ai.retrieval.web.exa import search_exa
from futureverse_ai.retrieval.web.perplexity import search_perplexity
from futureverse_ai.retrieval.web.sougou import search_sougou
from futureverse_ai.retrieval.web.firecrawl import search_firecrawl
from futureverse_ai.retrieval.web.external import search_external
from futureverse_ai.retrieval.web.yandex import search_yandex

from futureverse_ai.retrieval.utils import (
    get_content_from_url,
    get_embedding_function,
    get_reranking_function,
    get_model_path,
    query_collection,
    query_collection_with_hybrid_search,
    query_doc,
    query_doc_with_hybrid_search,
)
from futureverse_ai.retrieval.vector.utils import filter_metadata

from futureverse_ai.utils.misc import (
    calculate_sha256_string,
    sanitize_text_for_db,
)
from futureverse_ai.utils.auth import get_admin_user, get_verified_user


from futureverse_ai.utils.access_control import has_permission

from futureverse_ai.config import (
    ENV,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    UPLOAD_DIR,
    DEFAULT_LOCALE,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_QUERY_PREFIX,
)
from futureverse_ai.env import (
    DEVICE_TYPE,
    DOCKER,
    RAG_EMBEDDING_TIMEOUT,
    SENTENCE_TRANSFORMERS_BACKEND,
    SENTENCE_TRANSFORMERS_MODEL_KWARGS,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_BACKEND,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_MODEL_KWARGS,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_SIGMOID_ACTIVATION_FUNCTION,
)

from futureverse_ai.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)

##########################################
#
# Utility functions
#
##########################################


def get_ef(
    engine: str,
    embedding_model: str,
    auto_update: bool = RAG_EMBEDDING_MODEL_AUTO_UPDATE,
):
    ef = None
    # "" = local sentence-transformers; "sentence_transformers" is the DB-stored alias for the same engine
    if embedding_model and engine in ("", "sentence_transformers"):
        from sentence_transformers import SentenceTransformer

        model_path = get_model_path(embedding_model, auto_update)
        try:
            _st_mkwargs = dict(SENTENCE_TRANSFORMERS_MODEL_KWARGS or {})
            _st_mkwargs.pop("device", None)

            ef = SentenceTransformer(
                get_model_path(embedding_model, auto_update),
                device=DEVICE_TYPE,
                trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
                backend=SENTENCE_TRANSFORMERS_BACKEND,
                model_kwargs=_st_mkwargs or None,
            )
            log.info(f"SentenceTransformer loaded: model='{embedding_model}' device={DEVICE_TYPE}")
        except Exception as e:
            log.error(
                f"SentenceTransformer failed to load: model='{embedding_model}' "
                f"path='{model_path}' device={DEVICE_TYPE} — {e}"
            )

    return ef


def get_rf(
    engine: str = "",
    reranking_model: Optional[str] = None,
    external_reranker_url: str = "",
    external_reranker_api_key: str = "",
    external_reranker_timeout: str = "",
    auto_update: bool = RAG_RERANKING_MODEL_AUTO_UPDATE,
):
    rf = None
    # Convert timeout string to int or None (system default)
    timeout_value = (
        int(external_reranker_timeout) if external_reranker_timeout else None
    )
    if reranking_model:
        if any(model in reranking_model for model in ["jinaai/jina-colbert-v2"]):
            try:
                from futureverse_ai.retrieval.models.colbert import ColBERT

                rf = ColBERT(
                    get_model_path(reranking_model, auto_update),
                    env="docker" if DOCKER else None,
                )

            except Exception as e:
                log.error(f"ColBERT: {e}")
                raise Exception(ERROR_MESSAGES.DEFAULT(e))
        else:
            if engine == "external":
                try:
                    from futureverse_ai.retrieval.models.external import ExternalReranker

                    rf = ExternalReranker(
                        url=external_reranker_url,
                        api_key=external_reranker_api_key,
                        model=reranking_model,
                        timeout=timeout_value,
                    )
                except Exception as e:
                    log.error(f"ExternalReranking: {e}")
                    raise Exception(ERROR_MESSAGES.DEFAULT(e))
            else:
                import sentence_transformers
                import torch

                try:
                    # Strip "device" from model_kwargs — it's passed as a top-level
                    # CrossEncoder arg already. Keeping it causes TypeError on XLMRoberta.
                    _ce_mkwargs = dict(SENTENCE_TRANSFORMERS_CROSS_ENCODER_MODEL_KWARGS or {})
                    _ce_mkwargs.pop("device", None)
                    rf = sentence_transformers.CrossEncoder(
                        get_model_path(reranking_model, auto_update),
                        device=DEVICE_TYPE,
                        trust_remote_code=RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
                        backend=SENTENCE_TRANSFORMERS_CROSS_ENCODER_BACKEND,
                        model_kwargs=_ce_mkwargs or None,
                        activation_fn=(
                            torch.nn.Sigmoid()
                            if SENTENCE_TRANSFORMERS_CROSS_ENCODER_SIGMOID_ACTIVATION_FUNCTION
                            else None
                        ),
                    )
                except Exception as e:
                    log.error(f"CrossEncoder: {e}")
                    raise Exception(ERROR_MESSAGES.DEFAULT("CrossEncoder error"))

                # Safely adjust pad_token_id if missing as some models do not have this in config
                try:
                    model_cfg = getattr(rf, "model", None)
                    if model_cfg and hasattr(model_cfg, "config"):
                        cfg = model_cfg.config
                        if getattr(cfg, "pad_token_id", None) is None:
                            # Fallback to eos_token_id when available
                            eos = getattr(cfg, "eos_token_id", None)
                            if eos is not None:
                                cfg.pad_token_id = eos
                                log.debug(
                                    f"Missing pad_token_id detected; set to eos_token_id={eos}"
                                )
                            else:
                                log.warning(
                                    "Neither pad_token_id nor eos_token_id present in model config"
                                )
                except Exception as e2:
                    log.warning(f"Failed to adjust pad_token_id on CrossEncoder: {e2}")

    return rf


##########################################
#
# API routes
#
##########################################


router = APIRouter()


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = None


class ProcessUrlForm(CollectionNameForm):
    url: str


class SearchForm(BaseModel):
    queries: List[str]


@router.get("/")
async def get_status(request: Request):
    return {
        "status": True,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
        "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
    }


@router.get("/embedding")
async def get_embedding_config(request: Request, user=Depends(get_admin_user)):
    return {
        "status": True,
        "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
        "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
        "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        "openai_config": {
            "url": request.app.state.config.RAG_OPENAI_API_BASE_URL,
            "key": request.app.state.config.RAG_OPENAI_API_KEY,
        },
        "ollama_config": {
            "url": request.app.state.config.RAG_OLLAMA_BASE_URL,
            "key": request.app.state.config.RAG_OLLAMA_API_KEY,
        },
        "azure_openai_config": {
            "url": request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
            "key": request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
            "version": request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
        },
    }


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class OllamaConfigForm(BaseModel):
    url: str
    key: str


class AzureOpenAIConfigForm(BaseModel):
    url: str
    key: str
    version: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    ollama_config: Optional[OllamaConfigForm] = None
    azure_openai_config: Optional[AzureOpenAIConfigForm] = None
    RAG_EMBEDDING_ENGINE: str
    RAG_EMBEDDING_MODEL: str
    RAG_EMBEDDING_BATCH_SIZE: Optional[int] = 1
    ENABLE_ASYNC_EMBEDDING: Optional[bool] = True


def unload_embedding_model(request: Request):
    if request.app.state.config.RAG_EMBEDDING_ENGINE in ("", "sentence_transformers"):
        # unloads current internal embedding model and clears VRAM cache
        request.app.state.ef = None
        request.app.state.EMBEDDING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == "cuda":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@router.post("/embedding/update")
async def update_embedding_config(
    request: Request, form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating embedding model: {request.app.state.config.RAG_EMBEDDING_MODEL} to {form_data.RAG_EMBEDDING_MODEL}"
    )
    unload_embedding_model(request)
    try:
        request.app.state.config.RAG_EMBEDDING_ENGINE = form_data.RAG_EMBEDDING_ENGINE
        request.app.state.config.RAG_EMBEDDING_MODEL = form_data.RAG_EMBEDDING_MODEL
        request.app.state.config.RAG_EMBEDDING_BATCH_SIZE = (
            form_data.RAG_EMBEDDING_BATCH_SIZE
        )
        request.app.state.config.ENABLE_ASYNC_EMBEDDING = (
            form_data.ENABLE_ASYNC_EMBEDDING
        )

        if request.app.state.config.RAG_EMBEDDING_ENGINE in [
            "ollama",
            "openai",
            "azure_openai",
        ]:
            if form_data.openai_config is not None:
                request.app.state.config.RAG_OPENAI_API_BASE_URL = (
                    form_data.openai_config.url
                )
                request.app.state.config.RAG_OPENAI_API_KEY = (
                    form_data.openai_config.key
                )

            if form_data.ollama_config is not None:
                request.app.state.config.RAG_OLLAMA_BASE_URL = (
                    form_data.ollama_config.url
                )
                request.app.state.config.RAG_OLLAMA_API_KEY = (
                    form_data.ollama_config.key
                )

            if form_data.azure_openai_config is not None:
                request.app.state.config.RAG_AZURE_OPENAI_BASE_URL = (
                    form_data.azure_openai_config.url
                )
                request.app.state.config.RAG_AZURE_OPENAI_API_KEY = (
                    form_data.azure_openai_config.key
                )
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION = (
                    form_data.azure_openai_config.version
                )

        request.app.state.ef = get_ef(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
        )

        request.app.state.EMBEDDING_FUNCTION = get_embedding_function(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else (
                    request.app.state.config.RAG_OLLAMA_BASE_URL
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                    else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
                )
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else (
                    request.app.state.config.RAG_OLLAMA_API_KEY
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                    else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
                )
            ),
            request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "azure_openai"
                else None
            ),
            enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        )

        return {
            "status": True,
            "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
            "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
            "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
            "openai_config": {
                "url": request.app.state.config.RAG_OPENAI_API_BASE_URL,
                "key": request.app.state.config.RAG_OPENAI_API_KEY,
            },
            "ollama_config": {
                "url": request.app.state.config.RAG_OLLAMA_BASE_URL,
                "key": request.app.state.config.RAG_OLLAMA_API_KEY,
            },
            "azure_openai_config": {
                "url": request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
                "key": request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
                "version": request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
            },
        }
    except Exception as e:
        log.exception(f"Problem updating embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.get("/config")
async def get_rag_config(request: Request, user=Depends(get_admin_user)):
    return {
        "status": True,
        # RAG settings
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "TOP_K": request.app.state.config.TOP_K,
        "BYPASS_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        "RAG_FULL_CONTEXT": request.app.state.config.RAG_FULL_CONTEXT,
        # Neo4j knowledge graph settings
        "ENABLE_NEO4J": request.app.state.config.ENABLE_NEO4J,
        # Hybrid search settings
        "ENABLE_RAG_HYBRID_SEARCH": request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
        "ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS": request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS,
        "TOP_K_RERANKER": request.app.state.config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": request.app.state.config.RELEVANCE_THRESHOLD,
        "HYBRID_BM25_WEIGHT": request.app.state.config.HYBRID_BM25_WEIGHT,
        # Content extraction settings
        "CONTENT_EXTRACTION_ENGINE": request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        "PDF_EXTRACT_IMAGES": request.app.state.config.PDF_EXTRACT_IMAGES,
        "PDF_LOADER_MODE": request.app.state.config.PDF_LOADER_MODE,
        "DATALAB_MARKER_API_KEY": request.app.state.config.DATALAB_MARKER_API_KEY,
        "DATALAB_MARKER_API_BASE_URL": request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        "DATALAB_MARKER_ADDITIONAL_CONFIG": request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        "DATALAB_MARKER_SKIP_CACHE": request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        "DATALAB_MARKER_FORCE_OCR": request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        "DATALAB_MARKER_PAGINATE": request.app.state.config.DATALAB_MARKER_PAGINATE,
        "DATALAB_MARKER_STRIP_EXISTING_OCR": request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        "DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION": request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        "DATALAB_MARKER_FORMAT_LINES": request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
        "DATALAB_MARKER_USE_LLM": request.app.state.config.DATALAB_MARKER_USE_LLM,
        "DATALAB_MARKER_OUTPUT_FORMAT": request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        "EXTERNAL_DOCUMENT_LOADER_URL": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        "EXTERNAL_DOCUMENT_LOADER_API_KEY": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        "TIKA_SERVER_URL": request.app.state.config.TIKA_SERVER_URL,
        "DOCLING_SERVER_URL": request.app.state.config.DOCLING_SERVER_URL,
        "DOCLING_API_KEY": request.app.state.config.DOCLING_API_KEY,
        "DOCLING_PARAMS": request.app.state.config.DOCLING_PARAMS,
        "DOCUMENT_INTELLIGENCE_ENDPOINT": request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        "DOCUMENT_INTELLIGENCE_KEY": request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        "DOCUMENT_INTELLIGENCE_MODEL": request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        "MISTRAL_OCR_API_BASE_URL": request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        "MISTRAL_OCR_API_KEY": request.app.state.config.MISTRAL_OCR_API_KEY,
        # MinerU settings
        "MINERU_API_MODE": request.app.state.config.MINERU_API_MODE,
        "MINERU_API_URL": request.app.state.config.MINERU_API_URL,
        "MINERU_API_KEY": request.app.state.config.MINERU_API_KEY,
        "MINERU_API_TIMEOUT": request.app.state.config.MINERU_API_TIMEOUT,
        "MINERU_PARAMS": request.app.state.config.MINERU_PARAMS,
        # Reranking settings
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_RERANKING_ENGINE": request.app.state.config.RAG_RERANKING_ENGINE,
        "RAG_EXTERNAL_RERANKER_URL": request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        "RAG_EXTERNAL_RERANKER_API_KEY": request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        "RAG_EXTERNAL_RERANKER_TIMEOUT": request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        # Chunking settings
        "CHUNKING_STRATEGY": request.app.state.config.CHUNKING_STRATEGY,
        "PARAGRAPH_MIN_SENTENCES": request.app.state.config.PARAGRAPH_MIN_SENTENCES,
        "PARAGRAPH_MAX_SENTENCES": request.app.state.config.PARAGRAPH_MAX_SENTENCES,
        "ENABLE_KEYWORD_EXTRACTION": request.app.state.config.ENABLE_KEYWORD_EXTRACTION,
        "KEYWORD_EXTRACTION_METHOD": request.app.state.config.KEYWORD_EXTRACTION_METHOD,
        "TEXT_SPLITTER": request.app.state.config.TEXT_SPLITTER,
        "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER,
        "ENABLE_CONTEXTUAL_CHUNKING": request.app.state.config.ENABLE_CONTEXTUAL_CHUNKING,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_MIN_SIZE_TARGET": request.app.state.config.CHUNK_MIN_SIZE_TARGET,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        "FILE_MAX_SIZE": request.app.state.config.FILE_MAX_SIZE,
        "FILE_MAX_COUNT": request.app.state.config.FILE_MAX_COUNT,
        "FILE_IMAGE_COMPRESSION_WIDTH": request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        "FILE_IMAGE_COMPRESSION_HEIGHT": request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        "ALLOWED_FILE_EXTENSIONS": request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        "ENABLE_GOOGLE_DRIVE_INTEGRATION": request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        "ENABLE_ONEDRIVE_INTEGRATION": request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings
        "web": {
            "ENABLE_WEB_SEARCH": request.app.state.config.ENABLE_WEB_SEARCH,
            "WEB_SEARCH_ENGINE": request.app.state.config.WEB_SEARCH_ENGINE,
            "WEB_SEARCH_TRUST_ENV": request.app.state.config.WEB_SEARCH_TRUST_ENV,
            "WEB_SEARCH_RESULT_COUNT": request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            "WEB_SEARCH_CONCURRENT_REQUESTS": request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            "WEB_LOADER_CONCURRENT_REQUESTS": request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            "WEB_SEARCH_DOMAIN_FILTER_LIST": request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            "BYPASS_WEB_SEARCH_WEB_LOADER": request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            "OLLAMA_CLOUD_WEB_SEARCH_API_KEY": request.app.state.config.OLLAMA_CLOUD_WEB_SEARCH_API_KEY,
            "SEARXNG_QUERY_URL": request.app.state.config.SEARXNG_QUERY_URL,
            "SEARXNG_LANGUAGE": request.app.state.config.SEARXNG_LANGUAGE,
            "YACY_QUERY_URL": request.app.state.config.YACY_QUERY_URL,
            "YACY_USERNAME": request.app.state.config.YACY_USERNAME,
            "YACY_PASSWORD": request.app.state.config.YACY_PASSWORD,
            "GOOGLE_PSE_API_KEY": request.app.state.config.GOOGLE_PSE_API_KEY,
            "GOOGLE_PSE_ENGINE_ID": request.app.state.config.GOOGLE_PSE_ENGINE_ID,
            "BRAVE_SEARCH_API_KEY": request.app.state.config.BRAVE_SEARCH_API_KEY,
            "KAGI_SEARCH_API_KEY": request.app.state.config.KAGI_SEARCH_API_KEY,
            "MOJEEK_SEARCH_API_KEY": request.app.state.config.MOJEEK_SEARCH_API_KEY,
            "BOCHA_SEARCH_API_KEY": request.app.state.config.BOCHA_SEARCH_API_KEY,
            "SERPSTACK_API_KEY": request.app.state.config.SERPSTACK_API_KEY,
            "SERPSTACK_HTTPS": request.app.state.config.SERPSTACK_HTTPS,
            "SERPER_API_KEY": request.app.state.config.SERPER_API_KEY,
            "SERPLY_API_KEY": request.app.state.config.SERPLY_API_KEY,
            "DDGS_BACKEND": request.app.state.config.DDGS_BACKEND,
            "TAVILY_API_KEY": request.app.state.config.TAVILY_API_KEY,
            "SEARCHAPI_API_KEY": request.app.state.config.SEARCHAPI_API_KEY,
            "SEARCHAPI_ENGINE": request.app.state.config.SEARCHAPI_ENGINE,
            "SERPAPI_API_KEY": request.app.state.config.SERPAPI_API_KEY,
            "SERPAPI_ENGINE": request.app.state.config.SERPAPI_ENGINE,
            "JINA_API_KEY": request.app.state.config.JINA_API_KEY,
            "JINA_API_BASE_URL": request.app.state.config.JINA_API_BASE_URL,
            "BING_SEARCH_V7_ENDPOINT": request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            "BING_SEARCH_V7_SUBSCRIPTION_KEY": request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            "EXA_API_KEY": request.app.state.config.EXA_API_KEY,
            "PERPLEXITY_API_KEY": request.app.state.config.PERPLEXITY_API_KEY,
            "PERPLEXITY_MODEL": request.app.state.config.PERPLEXITY_MODEL,
            "PERPLEXITY_SEARCH_CONTEXT_USAGE": request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
            "PERPLEXITY_SEARCH_API_URL": request.app.state.config.PERPLEXITY_SEARCH_API_URL,
            "SOUGOU_API_SID": request.app.state.config.SOUGOU_API_SID,
            "SOUGOU_API_SK": request.app.state.config.SOUGOU_API_SK,
            "WEB_LOADER_ENGINE": request.app.state.config.WEB_LOADER_ENGINE,
            "WEB_LOADER_TIMEOUT": request.app.state.config.WEB_LOADER_TIMEOUT,
            "ENABLE_WEB_LOADER_SSL_VERIFICATION": request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            "PLAYWRIGHT_WS_URL": request.app.state.config.PLAYWRIGHT_WS_URL,
            "PLAYWRIGHT_TIMEOUT": request.app.state.config.PLAYWRIGHT_TIMEOUT,
            "FIRECRAWL_API_KEY": request.app.state.config.FIRECRAWL_API_KEY,
            "FIRECRAWL_API_BASE_URL": request.app.state.config.FIRECRAWL_API_BASE_URL,
            "FIRECRAWL_TIMEOUT": request.app.state.config.FIRECRAWL_TIMEOUT,
            "TAVILY_EXTRACT_DEPTH": request.app.state.config.TAVILY_EXTRACT_DEPTH,
            "EXTERNAL_WEB_SEARCH_URL": request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            "EXTERNAL_WEB_SEARCH_API_KEY": request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            "EXTERNAL_WEB_LOADER_URL": request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            "EXTERNAL_WEB_LOADER_API_KEY": request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            "YOUTUBE_LOADER_LANGUAGE": request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "YOUTUBE_LOADER_PROXY_URL": request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            "YOUTUBE_LOADER_TRANSLATION": request.app.state.YOUTUBE_LOADER_TRANSLATION,
            "YANDEX_WEB_SEARCH_URL": request.app.state.config.YANDEX_WEB_SEARCH_URL,
            "YANDEX_WEB_SEARCH_API_KEY": request.app.state.config.YANDEX_WEB_SEARCH_API_KEY,
            "YANDEX_WEB_SEARCH_CONFIG": request.app.state.config.YANDEX_WEB_SEARCH_CONFIG,
        },
    }


class WebConfig(BaseModel):
    ENABLE_WEB_SEARCH: Optional[bool] = None
    WEB_SEARCH_ENGINE: Optional[str] = None
    WEB_SEARCH_TRUST_ENV: Optional[bool] = None
    WEB_SEARCH_RESULT_COUNT: Optional[int] = None
    WEB_SEARCH_CONCURRENT_REQUESTS: Optional[int] = None
    WEB_LOADER_CONCURRENT_REQUESTS: Optional[int] = None
    WEB_SEARCH_DOMAIN_FILTER_LIST: Optional[List[str]] = []
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL: Optional[bool] = None
    BYPASS_WEB_SEARCH_WEB_LOADER: Optional[bool] = None
    OLLAMA_CLOUD_WEB_SEARCH_API_KEY: Optional[str] = None
    SEARXNG_QUERY_URL: Optional[str] = None
    SEARXNG_LANGUAGE: Optional[str] = None
    YACY_QUERY_URL: Optional[str] = None
    YACY_USERNAME: Optional[str] = None
    YACY_PASSWORD: Optional[str] = None
    GOOGLE_PSE_API_KEY: Optional[str] = None
    GOOGLE_PSE_ENGINE_ID: Optional[str] = None
    BRAVE_SEARCH_API_KEY: Optional[str] = None
    KAGI_SEARCH_API_KEY: Optional[str] = None
    MOJEEK_SEARCH_API_KEY: Optional[str] = None
    BOCHA_SEARCH_API_KEY: Optional[str] = None
    SERPSTACK_API_KEY: Optional[str] = None
    SERPSTACK_HTTPS: Optional[bool] = None
    SERPER_API_KEY: Optional[str] = None
    SERPLY_API_KEY: Optional[str] = None
    DDGS_BACKEND: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    SEARCHAPI_API_KEY: Optional[str] = None
    SEARCHAPI_ENGINE: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None
    SERPAPI_ENGINE: Optional[str] = None
    JINA_API_KEY: Optional[str] = None
    JINA_API_BASE_URL: Optional[str] = None
    BING_SEARCH_V7_ENDPOINT: Optional[str] = None
    BING_SEARCH_V7_SUBSCRIPTION_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    PERPLEXITY_MODEL: Optional[str] = None
    PERPLEXITY_SEARCH_CONTEXT_USAGE: Optional[str] = None
    PERPLEXITY_SEARCH_API_URL: Optional[str] = None
    SOUGOU_API_SID: Optional[str] = None
    SOUGOU_API_SK: Optional[str] = None
    WEB_LOADER_ENGINE: Optional[str] = None
    WEB_LOADER_TIMEOUT: Optional[str] = None
    ENABLE_WEB_LOADER_SSL_VERIFICATION: Optional[bool] = None
    PLAYWRIGHT_WS_URL: Optional[str] = None
    PLAYWRIGHT_TIMEOUT: Optional[int] = None
    FIRECRAWL_API_KEY: Optional[str] = None
    FIRECRAWL_API_BASE_URL: Optional[str] = None
    FIRECRAWL_TIMEOUT: Optional[str] = None
    TAVILY_EXTRACT_DEPTH: Optional[str] = None
    EXTERNAL_WEB_SEARCH_URL: Optional[str] = None
    EXTERNAL_WEB_SEARCH_API_KEY: Optional[str] = None
    EXTERNAL_WEB_LOADER_URL: Optional[str] = None
    EXTERNAL_WEB_LOADER_API_KEY: Optional[str] = None
    YOUTUBE_LOADER_LANGUAGE: Optional[List[str]] = None
    YOUTUBE_LOADER_PROXY_URL: Optional[str] = None
    YOUTUBE_LOADER_TRANSLATION: Optional[str] = None
    YANDEX_WEB_SEARCH_URL: Optional[str] = None
    YANDEX_WEB_SEARCH_API_KEY: Optional[str] = None
    YANDEX_WEB_SEARCH_CONFIG: Optional[str] = None


class ConfigForm(BaseModel):
    # RAG settings
    RAG_TEMPLATE: Optional[str] = None
    TOP_K: Optional[int] = None
    BYPASS_EMBEDDING_AND_RETRIEVAL: Optional[bool] = None
    RAG_FULL_CONTEXT: Optional[bool] = None

    # Neo4j knowledge graph settings
    ENABLE_NEO4J: Optional[bool] = None

    # Hybrid search settings
    ENABLE_RAG_HYBRID_SEARCH: Optional[bool] = None
    ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS: Optional[bool] = None
    TOP_K_RERANKER: Optional[int] = None
    RELEVANCE_THRESHOLD: Optional[float] = None
    HYBRID_BM25_WEIGHT: Optional[float] = None

    # Content extraction settings
    CONTENT_EXTRACTION_ENGINE: Optional[str] = None
    PDF_EXTRACT_IMAGES: Optional[bool] = None
    PDF_LOADER_MODE: Optional[str] = None

    DATALAB_MARKER_API_KEY: Optional[str] = None
    DATALAB_MARKER_API_BASE_URL: Optional[str] = None
    DATALAB_MARKER_ADDITIONAL_CONFIG: Optional[str] = None
    DATALAB_MARKER_SKIP_CACHE: Optional[bool] = None
    DATALAB_MARKER_FORCE_OCR: Optional[bool] = None
    DATALAB_MARKER_PAGINATE: Optional[bool] = None
    DATALAB_MARKER_STRIP_EXISTING_OCR: Optional[bool] = None
    DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION: Optional[bool] = None
    DATALAB_MARKER_FORMAT_LINES: Optional[bool] = None
    DATALAB_MARKER_USE_LLM: Optional[bool] = None
    DATALAB_MARKER_OUTPUT_FORMAT: Optional[str] = None

    EXTERNAL_DOCUMENT_LOADER_URL: Optional[str] = None
    EXTERNAL_DOCUMENT_LOADER_API_KEY: Optional[str] = None

    TIKA_SERVER_URL: Optional[str] = None
    DOCLING_SERVER_URL: Optional[str] = None
    DOCLING_API_KEY: Optional[str] = None
    DOCLING_PARAMS: Optional[dict] = None
    DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = None
    DOCUMENT_INTELLIGENCE_KEY: Optional[str] = None
    DOCUMENT_INTELLIGENCE_MODEL: Optional[str] = None
    MISTRAL_OCR_API_BASE_URL: Optional[str] = None
    MISTRAL_OCR_API_KEY: Optional[str] = None

    # MinerU settings
    MINERU_API_MODE: Optional[str] = None
    MINERU_API_URL: Optional[str] = None
    MINERU_API_KEY: Optional[str] = None
    MINERU_API_TIMEOUT: Optional[str] = None
    MINERU_PARAMS: Optional[dict] = None

    # Reranking settings
    RAG_RERANKING_MODEL: Optional[str] = None
    RAG_RERANKING_ENGINE: Optional[str] = None
    RAG_EXTERNAL_RERANKER_URL: Optional[str] = None
    RAG_EXTERNAL_RERANKER_API_KEY: Optional[str] = None
    RAG_EXTERNAL_RERANKER_TIMEOUT: Optional[str] = None

    # Chunking settings
    CHUNKING_STRATEGY: Optional[str] = None
    PARAGRAPH_MIN_SENTENCES: Optional[int] = None
    PARAGRAPH_MAX_SENTENCES: Optional[int] = None
    ENABLE_KEYWORD_EXTRACTION: Optional[bool] = None
    KEYWORD_EXTRACTION_METHOD: Optional[str] = None
    TEXT_SPLITTER: Optional[str] = None
    ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER: Optional[bool] = None
    ENABLE_CONTEXTUAL_CHUNKING: Optional[bool] = None
    CHUNK_SIZE: Optional[int] = None
    CHUNK_MIN_SIZE_TARGET: Optional[int] = None
    CHUNK_OVERLAP: Optional[int] = None

    # File upload settings
    FILE_MAX_SIZE: Optional[int] = None
    FILE_MAX_COUNT: Optional[int] = None
    FILE_IMAGE_COMPRESSION_WIDTH: Optional[int] = None
    FILE_IMAGE_COMPRESSION_HEIGHT: Optional[int] = None
    ALLOWED_FILE_EXTENSIONS: Optional[List[str]] = None

    # Integration settings
    ENABLE_GOOGLE_DRIVE_INTEGRATION: Optional[bool] = None
    ENABLE_ONEDRIVE_INTEGRATION: Optional[bool] = None

    # Web search settings
    web: Optional[WebConfig] = None


@router.post("/config/update")
async def update_rag_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    # RAG settings
    request.app.state.config.RAG_TEMPLATE = (
        form_data.RAG_TEMPLATE
        if form_data.RAG_TEMPLATE is not None
        else request.app.state.config.RAG_TEMPLATE
    )
    request.app.state.config.TOP_K = (
        form_data.TOP_K
        if form_data.TOP_K is not None
        else request.app.state.config.TOP_K
    )
    request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = (
        form_data.BYPASS_EMBEDDING_AND_RETRIEVAL
        if form_data.BYPASS_EMBEDDING_AND_RETRIEVAL is not None
        else request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
    )
    request.app.state.config.RAG_FULL_CONTEXT = (
        form_data.RAG_FULL_CONTEXT
        if form_data.RAG_FULL_CONTEXT is not None
        else request.app.state.config.RAG_FULL_CONTEXT
    )

    # Neo4j knowledge graph settings
    request.app.state.config.ENABLE_NEO4J = (
        form_data.ENABLE_NEO4J
        if form_data.ENABLE_NEO4J is not None
        else request.app.state.config.ENABLE_NEO4J
    )

    # Hybrid search settings
    request.app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.ENABLE_RAG_HYBRID_SEARCH
        if form_data.ENABLE_RAG_HYBRID_SEARCH is not None
        else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH
    )
    request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS = (
        form_data.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
        if form_data.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS is not None
        else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
    )

    request.app.state.config.TOP_K_RERANKER = (
        form_data.TOP_K_RERANKER
        if form_data.TOP_K_RERANKER is not None
        else request.app.state.config.TOP_K_RERANKER
    )
    request.app.state.config.RELEVANCE_THRESHOLD = (
        form_data.RELEVANCE_THRESHOLD
        if form_data.RELEVANCE_THRESHOLD is not None
        else request.app.state.config.RELEVANCE_THRESHOLD
    )
    request.app.state.config.HYBRID_BM25_WEIGHT = (
        form_data.HYBRID_BM25_WEIGHT
        if form_data.HYBRID_BM25_WEIGHT is not None
        else request.app.state.config.HYBRID_BM25_WEIGHT
    )

    # Content extraction settings
    request.app.state.config.CONTENT_EXTRACTION_ENGINE = (
        form_data.CONTENT_EXTRACTION_ENGINE
        if form_data.CONTENT_EXTRACTION_ENGINE is not None
        else request.app.state.config.CONTENT_EXTRACTION_ENGINE
    )
    request.app.state.config.PDF_EXTRACT_IMAGES = (
        form_data.PDF_EXTRACT_IMAGES
        if form_data.PDF_EXTRACT_IMAGES is not None
        else request.app.state.config.PDF_EXTRACT_IMAGES
    )
    request.app.state.config.PDF_LOADER_MODE = (
        form_data.PDF_LOADER_MODE
        if form_data.PDF_LOADER_MODE is not None
        else request.app.state.config.PDF_LOADER_MODE
    )
    request.app.state.config.DATALAB_MARKER_API_KEY = (
        form_data.DATALAB_MARKER_API_KEY
        if form_data.DATALAB_MARKER_API_KEY is not None
        else request.app.state.config.DATALAB_MARKER_API_KEY
    )
    request.app.state.config.DATALAB_MARKER_API_BASE_URL = (
        form_data.DATALAB_MARKER_API_BASE_URL
        if form_data.DATALAB_MARKER_API_BASE_URL is not None
        else request.app.state.config.DATALAB_MARKER_API_BASE_URL
    )
    request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG = (
        form_data.DATALAB_MARKER_ADDITIONAL_CONFIG
        if form_data.DATALAB_MARKER_ADDITIONAL_CONFIG is not None
        else request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG
    )
    request.app.state.config.DATALAB_MARKER_SKIP_CACHE = (
        form_data.DATALAB_MARKER_SKIP_CACHE
        if form_data.DATALAB_MARKER_SKIP_CACHE is not None
        else request.app.state.config.DATALAB_MARKER_SKIP_CACHE
    )
    request.app.state.config.DATALAB_MARKER_FORCE_OCR = (
        form_data.DATALAB_MARKER_FORCE_OCR
        if form_data.DATALAB_MARKER_FORCE_OCR is not None
        else request.app.state.config.DATALAB_MARKER_FORCE_OCR
    )
    request.app.state.config.DATALAB_MARKER_PAGINATE = (
        form_data.DATALAB_MARKER_PAGINATE
        if form_data.DATALAB_MARKER_PAGINATE is not None
        else request.app.state.config.DATALAB_MARKER_PAGINATE
    )
    request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR = (
        form_data.DATALAB_MARKER_STRIP_EXISTING_OCR
        if form_data.DATALAB_MARKER_STRIP_EXISTING_OCR is not None
        else request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR
    )
    request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION = (
        form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
        if form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION is not None
        else request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
    )
    request.app.state.config.DATALAB_MARKER_FORMAT_LINES = (
        form_data.DATALAB_MARKER_FORMAT_LINES
        if form_data.DATALAB_MARKER_FORMAT_LINES is not None
        else request.app.state.config.DATALAB_MARKER_FORMAT_LINES
    )
    request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT = (
        form_data.DATALAB_MARKER_OUTPUT_FORMAT
        if form_data.DATALAB_MARKER_OUTPUT_FORMAT is not None
        else request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT
    )
    request.app.state.config.DATALAB_MARKER_USE_LLM = (
        form_data.DATALAB_MARKER_USE_LLM
        if form_data.DATALAB_MARKER_USE_LLM is not None
        else request.app.state.config.DATALAB_MARKER_USE_LLM
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL = (
        form_data.EXTERNAL_DOCUMENT_LOADER_URL
        if form_data.EXTERNAL_DOCUMENT_LOADER_URL is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY = (
        form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY
        if form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY
    )
    request.app.state.config.TIKA_SERVER_URL = (
        form_data.TIKA_SERVER_URL
        if form_data.TIKA_SERVER_URL is not None
        else request.app.state.config.TIKA_SERVER_URL
    )
    request.app.state.config.DOCLING_SERVER_URL = (
        form_data.DOCLING_SERVER_URL
        if form_data.DOCLING_SERVER_URL is not None
        else request.app.state.config.DOCLING_SERVER_URL
    )
    request.app.state.config.DOCLING_API_KEY = (
        form_data.DOCLING_API_KEY
        if form_data.DOCLING_API_KEY is not None
        else request.app.state.config.DOCLING_API_KEY
    )
    request.app.state.config.DOCLING_PARAMS = (
        form_data.DOCLING_PARAMS
        if form_data.DOCLING_PARAMS is not None
        else request.app.state.config.DOCLING_PARAMS
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = (
        form_data.DOCUMENT_INTELLIGENCE_ENDPOINT
        if form_data.DOCUMENT_INTELLIGENCE_ENDPOINT is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_KEY = (
        form_data.DOCUMENT_INTELLIGENCE_KEY
        if form_data.DOCUMENT_INTELLIGENCE_KEY is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_KEY
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL = (
        form_data.DOCUMENT_INTELLIGENCE_MODEL
        if form_data.DOCUMENT_INTELLIGENCE_MODEL is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL
    )

    request.app.state.config.MISTRAL_OCR_API_BASE_URL = (
        form_data.MISTRAL_OCR_API_BASE_URL
        if form_data.MISTRAL_OCR_API_BASE_URL is not None
        else request.app.state.config.MISTRAL_OCR_API_BASE_URL
    )
    request.app.state.config.MISTRAL_OCR_API_KEY = (
        form_data.MISTRAL_OCR_API_KEY
        if form_data.MISTRAL_OCR_API_KEY is not None
        else request.app.state.config.MISTRAL_OCR_API_KEY
    )

    # MinerU settings
    request.app.state.config.MINERU_API_MODE = (
        form_data.MINERU_API_MODE
        if form_data.MINERU_API_MODE is not None
        else request.app.state.config.MINERU_API_MODE
    )
    request.app.state.config.MINERU_API_URL = (
        form_data.MINERU_API_URL
        if form_data.MINERU_API_URL is not None
        else request.app.state.config.MINERU_API_URL
    )
    request.app.state.config.MINERU_API_KEY = (
        form_data.MINERU_API_KEY
        if form_data.MINERU_API_KEY is not None
        else request.app.state.config.MINERU_API_KEY
    )
    request.app.state.config.MINERU_API_TIMEOUT = (
        form_data.MINERU_API_TIMEOUT
        if form_data.MINERU_API_TIMEOUT is not None
        else request.app.state.config.MINERU_API_TIMEOUT
    )
    request.app.state.config.MINERU_PARAMS = (
        form_data.MINERU_PARAMS
        if form_data.MINERU_PARAMS is not None
        else request.app.state.config.MINERU_PARAMS
    )

    # Reranking settings
    if request.app.state.config.RAG_RERANKING_ENGINE == "":
        # Unloading the internal reranker and clear VRAM memory
        request.app.state.rf = None
        request.app.state.RERANKING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == "cuda":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    request.app.state.config.RAG_RERANKING_ENGINE = (
        form_data.RAG_RERANKING_ENGINE
        if form_data.RAG_RERANKING_ENGINE is not None
        else request.app.state.config.RAG_RERANKING_ENGINE
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_URL = (
        form_data.RAG_EXTERNAL_RERANKER_URL
        if form_data.RAG_EXTERNAL_RERANKER_URL is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_URL
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY = (
        form_data.RAG_EXTERNAL_RERANKER_API_KEY
        if form_data.RAG_EXTERNAL_RERANKER_API_KEY is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT = (
        form_data.RAG_EXTERNAL_RERANKER_TIMEOUT
        if form_data.RAG_EXTERNAL_RERANKER_TIMEOUT is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT
    )

    log.info(
        f"Updating reranking model: {request.app.state.config.RAG_RERANKING_MODEL} to {form_data.RAG_RERANKING_MODEL}"
    )
    try:
        request.app.state.config.RAG_RERANKING_MODEL = (
            form_data.RAG_RERANKING_MODEL
            if form_data.RAG_RERANKING_MODEL is not None
            else request.app.state.config.RAG_RERANKING_MODEL
        )

        # Load reranking model if hybrid search is enabled.
        # If no reranking model is configured, rf stays None and
        # RerankCompressor falls back to cosine similarity reranking —
        # hybrid search still works without a dedicated reranker.
        # NEVER set ENABLE_RAG_HYBRID_SEARCH=False here; let the user's
        # saved preference persist regardless of reranker availability.
        if (
            request.app.state.config.ENABLE_RAG_HYBRID_SEARCH
            and not request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
        ):
            try:
                request.app.state.rf = get_rf(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
                )
                request.app.state.RERANKING_FUNCTION = get_reranking_function(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.rf,
                )
            except Exception as e:
                log.warning(
                    f"Reranking model could not be loaded ({e}); "
                    "hybrid search will use cosine similarity reranking instead."
                )
                request.app.state.rf = None
    except Exception as e:
        log.exception(f"Problem updating reranking model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

    # Chunking settings
    request.app.state.config.TEXT_SPLITTER = (
        form_data.TEXT_SPLITTER
        if form_data.TEXT_SPLITTER is not None
        else request.app.state.config.TEXT_SPLITTER
    )
    request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER = (
        form_data.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER
        if form_data.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER is not None
        else request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER
    )
    request.app.state.config.ENABLE_CONTEXTUAL_CHUNKING = (
        form_data.ENABLE_CONTEXTUAL_CHUNKING
        if form_data.ENABLE_CONTEXTUAL_CHUNKING is not None
        else request.app.state.config.ENABLE_CONTEXTUAL_CHUNKING
    )
    request.app.state.config.ENABLE_KEYWORD_EXTRACTION = (
        form_data.ENABLE_KEYWORD_EXTRACTION
        if form_data.ENABLE_KEYWORD_EXTRACTION is not None
        else request.app.state.config.ENABLE_KEYWORD_EXTRACTION
    )
    request.app.state.config.KEYWORD_EXTRACTION_METHOD = (
        form_data.KEYWORD_EXTRACTION_METHOD
        if form_data.KEYWORD_EXTRACTION_METHOD is not None
        else request.app.state.config.KEYWORD_EXTRACTION_METHOD
    )
    request.app.state.config.CHUNKING_STRATEGY = (
        form_data.CHUNKING_STRATEGY
        if form_data.CHUNKING_STRATEGY is not None
        else request.app.state.config.CHUNKING_STRATEGY
    )
    request.app.state.config.PARAGRAPH_MIN_SENTENCES = (
        form_data.PARAGRAPH_MIN_SENTENCES
        if form_data.PARAGRAPH_MIN_SENTENCES is not None
        else request.app.state.config.PARAGRAPH_MIN_SENTENCES
    )
    request.app.state.config.PARAGRAPH_MAX_SENTENCES = (
        form_data.PARAGRAPH_MAX_SENTENCES
        if form_data.PARAGRAPH_MAX_SENTENCES is not None
        else request.app.state.config.PARAGRAPH_MAX_SENTENCES
    )
    request.app.state.config.CHUNK_SIZE = (
        form_data.CHUNK_SIZE
        if form_data.CHUNK_SIZE is not None
        else request.app.state.config.CHUNK_SIZE
    )
    request.app.state.config.CHUNK_MIN_SIZE_TARGET = (
        form_data.CHUNK_MIN_SIZE_TARGET
        if form_data.CHUNK_MIN_SIZE_TARGET is not None
        else request.app.state.config.CHUNK_MIN_SIZE_TARGET
    )
    request.app.state.config.CHUNK_OVERLAP = (
        form_data.CHUNK_OVERLAP
        if form_data.CHUNK_OVERLAP is not None
        else request.app.state.config.CHUNK_OVERLAP
    )

    # File upload settings
    request.app.state.config.FILE_MAX_SIZE = (
        form_data.FILE_MAX_SIZE
        if form_data.FILE_MAX_SIZE is not None
        else request.app.state.config.FILE_MAX_SIZE
    )
    request.app.state.config.FILE_MAX_COUNT = (
        form_data.FILE_MAX_COUNT
        if form_data.FILE_MAX_COUNT is not None
        else request.app.state.config.FILE_MAX_COUNT
    )
    request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH = (
        form_data.FILE_IMAGE_COMPRESSION_WIDTH
        if form_data.FILE_IMAGE_COMPRESSION_WIDTH is not None
        else request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH
    )
    request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT = (
        form_data.FILE_IMAGE_COMPRESSION_HEIGHT
        if form_data.FILE_IMAGE_COMPRESSION_HEIGHT is not None
        else request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT
    )
    request.app.state.config.ALLOWED_FILE_EXTENSIONS = (
        form_data.ALLOWED_FILE_EXTENSIONS
        if form_data.ALLOWED_FILE_EXTENSIONS is not None
        else request.app.state.config.ALLOWED_FILE_EXTENSIONS
    )

    # Integration settings
    request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = (
        form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION
        if form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION
    )
    request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION = (
        form_data.ENABLE_ONEDRIVE_INTEGRATION
        if form_data.ENABLE_ONEDRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION
    )

    if form_data.web is not None:
        # Web search settings
        request.app.state.config.ENABLE_WEB_SEARCH = form_data.web.ENABLE_WEB_SEARCH
        request.app.state.config.WEB_SEARCH_ENGINE = form_data.web.WEB_SEARCH_ENGINE
        request.app.state.config.WEB_SEARCH_TRUST_ENV = (
            form_data.web.WEB_SEARCH_TRUST_ENV
        )
        request.app.state.config.WEB_SEARCH_RESULT_COUNT = (
            form_data.web.WEB_SEARCH_RESULT_COUNT
        )
        request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS = (
            form_data.web.WEB_SEARCH_CONCURRENT_REQUESTS
        )
        request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS = (
            form_data.web.WEB_LOADER_CONCURRENT_REQUESTS
        )
        request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST = (
            form_data.web.WEB_SEARCH_DOMAIN_FILTER_LIST
        )
        request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
            form_data.web.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
        )
        request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER = (
            form_data.web.BYPASS_WEB_SEARCH_WEB_LOADER
        )
        request.app.state.config.OLLAMA_CLOUD_WEB_SEARCH_API_KEY = (
            form_data.web.OLLAMA_CLOUD_WEB_SEARCH_API_KEY
        )
        request.app.state.config.SEARXNG_QUERY_URL = form_data.web.SEARXNG_QUERY_URL
        request.app.state.config.SEARXNG_LANGUAGE = form_data.web.SEARXNG_LANGUAGE
        request.app.state.config.YACY_QUERY_URL = form_data.web.YACY_QUERY_URL
        request.app.state.config.YACY_USERNAME = form_data.web.YACY_USERNAME
        request.app.state.config.YACY_PASSWORD = form_data.web.YACY_PASSWORD
        request.app.state.config.GOOGLE_PSE_API_KEY = form_data.web.GOOGLE_PSE_API_KEY
        request.app.state.config.GOOGLE_PSE_ENGINE_ID = (
            form_data.web.GOOGLE_PSE_ENGINE_ID
        )
        request.app.state.config.BRAVE_SEARCH_API_KEY = (
            form_data.web.BRAVE_SEARCH_API_KEY
        )
        request.app.state.config.KAGI_SEARCH_API_KEY = form_data.web.KAGI_SEARCH_API_KEY
        request.app.state.config.MOJEEK_SEARCH_API_KEY = (
            form_data.web.MOJEEK_SEARCH_API_KEY
        )
        request.app.state.config.BOCHA_SEARCH_API_KEY = (
            form_data.web.BOCHA_SEARCH_API_KEY
        )
        request.app.state.config.SERPSTACK_API_KEY = form_data.web.SERPSTACK_API_KEY
        request.app.state.config.SERPSTACK_HTTPS = form_data.web.SERPSTACK_HTTPS
        request.app.state.config.SERPER_API_KEY = form_data.web.SERPER_API_KEY
        request.app.state.config.SERPLY_API_KEY = form_data.web.SERPLY_API_KEY
        request.app.state.config.DDGS_BACKEND = form_data.web.DDGS_BACKEND
        request.app.state.config.TAVILY_API_KEY = form_data.web.TAVILY_API_KEY
        request.app.state.config.SEARCHAPI_API_KEY = form_data.web.SEARCHAPI_API_KEY
        request.app.state.config.SEARCHAPI_ENGINE = form_data.web.SEARCHAPI_ENGINE
        request.app.state.config.SERPAPI_API_KEY = form_data.web.SERPAPI_API_KEY
        request.app.state.config.SERPAPI_ENGINE = form_data.web.SERPAPI_ENGINE
        request.app.state.config.JINA_API_KEY = form_data.web.JINA_API_KEY
        request.app.state.config.JINA_API_BASE_URL = form_data.web.JINA_API_BASE_URL
        request.app.state.config.BING_SEARCH_V7_ENDPOINT = (
            form_data.web.BING_SEARCH_V7_ENDPOINT
        )
        request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = (
            form_data.web.BING_SEARCH_V7_SUBSCRIPTION_KEY
        )
        request.app.state.config.EXA_API_KEY = form_data.web.EXA_API_KEY
        request.app.state.config.PERPLEXITY_API_KEY = form_data.web.PERPLEXITY_API_KEY
        request.app.state.config.PERPLEXITY_MODEL = form_data.web.PERPLEXITY_MODEL
        request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE = (
            form_data.web.PERPLEXITY_SEARCH_CONTEXT_USAGE
        )
        request.app.state.config.PERPLEXITY_SEARCH_API_URL = (
            form_data.web.PERPLEXITY_SEARCH_API_URL
        )
        request.app.state.config.SOUGOU_API_SID = form_data.web.SOUGOU_API_SID
        request.app.state.config.SOUGOU_API_SK = form_data.web.SOUGOU_API_SK

        # Web loader settings
        request.app.state.config.WEB_LOADER_ENGINE = form_data.web.WEB_LOADER_ENGINE
        request.app.state.config.WEB_LOADER_TIMEOUT = form_data.web.WEB_LOADER_TIMEOUT

        request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION = (
            form_data.web.ENABLE_WEB_LOADER_SSL_VERIFICATION
        )
        request.app.state.config.PLAYWRIGHT_WS_URL = form_data.web.PLAYWRIGHT_WS_URL
        request.app.state.config.PLAYWRIGHT_TIMEOUT = form_data.web.PLAYWRIGHT_TIMEOUT
        request.app.state.config.FIRECRAWL_API_KEY = form_data.web.FIRECRAWL_API_KEY
        request.app.state.config.FIRECRAWL_API_BASE_URL = (
            form_data.web.FIRECRAWL_API_BASE_URL
        )
        request.app.state.config.FIRECRAWL_TIMEOUT = form_data.web.FIRECRAWL_TIMEOUT
        request.app.state.config.EXTERNAL_WEB_SEARCH_URL = (
            form_data.web.EXTERNAL_WEB_SEARCH_URL
        )
        request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY = (
            form_data.web.EXTERNAL_WEB_SEARCH_API_KEY
        )
        request.app.state.config.EXTERNAL_WEB_LOADER_URL = (
            form_data.web.EXTERNAL_WEB_LOADER_URL
        )
        request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY = (
            form_data.web.EXTERNAL_WEB_LOADER_API_KEY
        )
        request.app.state.config.TAVILY_EXTRACT_DEPTH = (
            form_data.web.TAVILY_EXTRACT_DEPTH
        )
        request.app.state.config.YOUTUBE_LOADER_LANGUAGE = (
            form_data.web.YOUTUBE_LOADER_LANGUAGE
        )
        request.app.state.config.YOUTUBE_LOADER_PROXY_URL = (
            form_data.web.YOUTUBE_LOADER_PROXY_URL
        )
        request.app.state.YOUTUBE_LOADER_TRANSLATION = (
            form_data.web.YOUTUBE_LOADER_TRANSLATION
        )
        request.app.state.config.YANDEX_WEB_SEARCH_URL = (
            form_data.web.YANDEX_WEB_SEARCH_URL
        )
        request.app.state.config.YANDEX_WEB_SEARCH_API_KEY = (
            form_data.web.YANDEX_WEB_SEARCH_API_KEY
        )
        request.app.state.config.YANDEX_WEB_SEARCH_CONFIG = (
            form_data.web.YANDEX_WEB_SEARCH_CONFIG
        )

    return {
        "status": True,
        # RAG settings
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "TOP_K": request.app.state.config.TOP_K,
        "BYPASS_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        "RAG_FULL_CONTEXT": request.app.state.config.RAG_FULL_CONTEXT,
        # Hybrid search settings
        "ENABLE_RAG_HYBRID_SEARCH": request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
        "TOP_K_RERANKER": request.app.state.config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": request.app.state.config.RELEVANCE_THRESHOLD,
        "HYBRID_BM25_WEIGHT": request.app.state.config.HYBRID_BM25_WEIGHT,
        # Content extraction settings
        "CONTENT_EXTRACTION_ENGINE": request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        "PDF_EXTRACT_IMAGES": request.app.state.config.PDF_EXTRACT_IMAGES,
        "PDF_LOADER_MODE": request.app.state.config.PDF_LOADER_MODE,
        "DATALAB_MARKER_API_KEY": request.app.state.config.DATALAB_MARKER_API_KEY,
        "DATALAB_MARKER_API_BASE_URL": request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        "DATALAB_MARKER_ADDITIONAL_CONFIG": request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        "DATALAB_MARKER_SKIP_CACHE": request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        "DATALAB_MARKER_FORCE_OCR": request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        "DATALAB_MARKER_PAGINATE": request.app.state.config.DATALAB_MARKER_PAGINATE,
        "DATALAB_MARKER_STRIP_EXISTING_OCR": request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        "DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION": request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        "DATALAB_MARKER_USE_LLM": request.app.state.config.DATALAB_MARKER_USE_LLM,
        "DATALAB_MARKER_OUTPUT_FORMAT": request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        "EXTERNAL_DOCUMENT_LOADER_URL": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        "EXTERNAL_DOCUMENT_LOADER_API_KEY": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        "TIKA_SERVER_URL": request.app.state.config.TIKA_SERVER_URL,
        "DOCLING_SERVER_URL": request.app.state.config.DOCLING_SERVER_URL,
        "DOCLING_API_KEY": request.app.state.config.DOCLING_API_KEY,
        "DOCLING_PARAMS": request.app.state.config.DOCLING_PARAMS,
        "DOCUMENT_INTELLIGENCE_ENDPOINT": request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        "DOCUMENT_INTELLIGENCE_KEY": request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        "DOCUMENT_INTELLIGENCE_MODEL": request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        "MISTRAL_OCR_API_BASE_URL": request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        "MISTRAL_OCR_API_KEY": request.app.state.config.MISTRAL_OCR_API_KEY,
        # MinerU settings
        "MINERU_API_MODE": request.app.state.config.MINERU_API_MODE,
        "MINERU_API_URL": request.app.state.config.MINERU_API_URL,
        "MINERU_API_KEY": request.app.state.config.MINERU_API_KEY,
        "MINERU_API_TIMEOUT": request.app.state.config.MINERU_API_TIMEOUT,
        "MINERU_PARAMS": request.app.state.config.MINERU_PARAMS,
        # Reranking settings
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_RERANKING_ENGINE": request.app.state.config.RAG_RERANKING_ENGINE,
        "RAG_EXTERNAL_RERANKER_URL": request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        "RAG_EXTERNAL_RERANKER_API_KEY": request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        "RAG_EXTERNAL_RERANKER_TIMEOUT": request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        # Chunking settings
        "TEXT_SPLITTER": request.app.state.config.TEXT_SPLITTER,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_MIN_SIZE_TARGET": request.app.state.config.CHUNK_MIN_SIZE_TARGET,
        "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        "FILE_MAX_SIZE": request.app.state.config.FILE_MAX_SIZE,
        "FILE_MAX_COUNT": request.app.state.config.FILE_MAX_COUNT,
        "FILE_IMAGE_COMPRESSION_WIDTH": request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        "FILE_IMAGE_COMPRESSION_HEIGHT": request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        "ALLOWED_FILE_EXTENSIONS": request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        "ENABLE_GOOGLE_DRIVE_INTEGRATION": request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        "ENABLE_ONEDRIVE_INTEGRATION": request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings
        "web": {
            "ENABLE_WEB_SEARCH": request.app.state.config.ENABLE_WEB_SEARCH,
            "WEB_SEARCH_ENGINE": request.app.state.config.WEB_SEARCH_ENGINE,
            "WEB_SEARCH_TRUST_ENV": request.app.state.config.WEB_SEARCH_TRUST_ENV,
            "WEB_SEARCH_RESULT_COUNT": request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            "WEB_SEARCH_CONCURRENT_REQUESTS": request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            "WEB_LOADER_CONCURRENT_REQUESTS": request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            "WEB_SEARCH_DOMAIN_FILTER_LIST": request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            "BYPASS_WEB_SEARCH_WEB_LOADER": request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            "OLLAMA_CLOUD_WEB_SEARCH_API_KEY": request.app.state.config.OLLAMA_CLOUD_WEB_SEARCH_API_KEY,
            "SEARXNG_QUERY_URL": request.app.state.config.SEARXNG_QUERY_URL,
            "SEARXNG_LANGUAGE": request.app.state.config.SEARXNG_LANGUAGE,
            "YACY_QUERY_URL": request.app.state.config.YACY_QUERY_URL,
            "YACY_USERNAME": request.app.state.config.YACY_USERNAME,
            "YACY_PASSWORD": request.app.state.config.YACY_PASSWORD,
            "GOOGLE_PSE_API_KEY": request.app.state.config.GOOGLE_PSE_API_KEY,
            "GOOGLE_PSE_ENGINE_ID": request.app.state.config.GOOGLE_PSE_ENGINE_ID,
            "BRAVE_SEARCH_API_KEY": request.app.state.config.BRAVE_SEARCH_API_KEY,
            "KAGI_SEARCH_API_KEY": request.app.state.config.KAGI_SEARCH_API_KEY,
            "MOJEEK_SEARCH_API_KEY": request.app.state.config.MOJEEK_SEARCH_API_KEY,
            "BOCHA_SEARCH_API_KEY": request.app.state.config.BOCHA_SEARCH_API_KEY,
            "SERPSTACK_API_KEY": request.app.state.config.SERPSTACK_API_KEY,
            "SERPSTACK_HTTPS": request.app.state.config.SERPSTACK_HTTPS,
            "SERPER_API_KEY": request.app.state.config.SERPER_API_KEY,
            "SERPLY_API_KEY": request.app.state.config.SERPLY_API_KEY,
            "TAVILY_API_KEY": request.app.state.config.TAVILY_API_KEY,
            "SEARCHAPI_API_KEY": request.app.state.config.SEARCHAPI_API_KEY,
            "SEARCHAPI_ENGINE": request.app.state.config.SEARCHAPI_ENGINE,
            "SERPAPI_API_KEY": request.app.state.config.SERPAPI_API_KEY,
            "SERPAPI_ENGINE": request.app.state.config.SERPAPI_ENGINE,
            "JINA_API_KEY": request.app.state.config.JINA_API_KEY,
            "JINA_API_BASE_URL": request.app.state.config.JINA_API_BASE_URL,
            "BING_SEARCH_V7_ENDPOINT": request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            "BING_SEARCH_V7_SUBSCRIPTION_KEY": request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            "EXA_API_KEY": request.app.state.config.EXA_API_KEY,
            "PERPLEXITY_API_KEY": request.app.state.config.PERPLEXITY_API_KEY,
            "PERPLEXITY_MODEL": request.app.state.config.PERPLEXITY_MODEL,
            "PERPLEXITY_SEARCH_CONTEXT_USAGE": request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
            "PERPLEXITY_SEARCH_API_URL": request.app.state.config.PERPLEXITY_SEARCH_API_URL,
            "SOUGOU_API_SID": request.app.state.config.SOUGOU_API_SID,
            "SOUGOU_API_SK": request.app.state.config.SOUGOU_API_SK,
            "WEB_LOADER_ENGINE": request.app.state.config.WEB_LOADER_ENGINE,
            "WEB_LOADER_TIMEOUT": request.app.state.config.WEB_LOADER_TIMEOUT,
            "ENABLE_WEB_LOADER_SSL_VERIFICATION": request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            "PLAYWRIGHT_WS_URL": request.app.state.config.PLAYWRIGHT_WS_URL,
            "PLAYWRIGHT_TIMEOUT": request.app.state.config.PLAYWRIGHT_TIMEOUT,
            "FIRECRAWL_API_KEY": request.app.state.config.FIRECRAWL_API_KEY,
            "FIRECRAWL_API_BASE_URL": request.app.state.config.FIRECRAWL_API_BASE_URL,
            "FIRECRAWL_TIMEOUT": request.app.state.config.FIRECRAWL_TIMEOUT,
            "TAVILY_EXTRACT_DEPTH": request.app.state.config.TAVILY_EXTRACT_DEPTH,
            "EXTERNAL_WEB_SEARCH_URL": request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            "EXTERNAL_WEB_SEARCH_API_KEY": request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            "EXTERNAL_WEB_LOADER_URL": request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            "EXTERNAL_WEB_LOADER_API_KEY": request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            "YOUTUBE_LOADER_LANGUAGE": request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "YOUTUBE_LOADER_PROXY_URL": request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            "YOUTUBE_LOADER_TRANSLATION": request.app.state.YOUTUBE_LOADER_TRANSLATION,
            "YANDEX_WEB_SEARCH_URL": request.app.state.config.YANDEX_WEB_SEARCH_URL,
            "YANDEX_WEB_SEARCH_API_KEY": request.app.state.config.YANDEX_WEB_SEARCH_API_KEY,
            "YANDEX_WEB_SEARCH_CONFIG": request.app.state.config.YANDEX_WEB_SEARCH_CONFIG,
        },
    }


####################################
#
# Document process and retrieval
#
####################################


def can_merge_chunks(a: Document, b: Document) -> bool:
    if a.metadata.get("source") != b.metadata.get("source"):
        return False

    a_file_id = a.metadata.get("file_id")
    b_file_id = b.metadata.get("file_id")

    if a_file_id is not None and b_file_id is not None:
        return a_file_id == b_file_id

    return True


def merge_docs_to_target_size(
    request: Request,
    chunks: list[Document],
) -> list[Document]:
    """
    Best-effort normalization of chunk sizes.

    Attempts to grow small chunks up to a desired minimum size,
    without exceeding the maximum size or crossing source/file
    boundaries.
    """
    min_chunk_size_target = request.app.state.config.CHUNK_MIN_SIZE_TARGET
    max_chunk_size = request.app.state.config.CHUNK_SIZE

    if min_chunk_size_target <= 0:
        return chunks

    measure_chunk_size = len
    if request.app.state.config.TEXT_SPLITTER == "token":
        encoding = tiktoken.get_encoding(
            str(request.app.state.config.TIKTOKEN_ENCODING_NAME)
        )
        measure_chunk_size = lambda text: len(encoding.encode(text))

    processed_chunks: list[Document] = []

    current_chunk: Document | None = None
    current_content: str = ""

    for next_chunk in chunks:
        if current_chunk is None:
            current_chunk = next_chunk
            current_content = next_chunk.page_content
            continue  # First chunk initialization

        proposed_content = f"{current_content}\n\n{next_chunk.page_content}"

        can_merge = (
            can_merge_chunks(current_chunk, next_chunk)
            and measure_chunk_size(current_content) < min_chunk_size_target
            and measure_chunk_size(proposed_content) <= max_chunk_size
        )

        if can_merge:
            current_content = proposed_content
        else:
            processed_chunks.append(
                Document(
                    page_content=current_content,
                    metadata={**current_chunk.metadata},
                )
            )
            current_chunk = next_chunk
            current_content = next_chunk.page_content

    if current_chunk is not None:
        processed_chunks.append(
            Document(
                page_content=current_content,
                metadata={**current_chunk.metadata},
            )
        )

    return processed_chunks


def save_docs_to_vector_db(
    request: Request,
    docs,
    collection_name,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None,
) -> bool:
    def _get_docs_info(docs: list[Document]) -> str:
        docs_info = set()
        for doc in docs:
            meta     = getattr(doc, "metadata", {})
            doc_name = (
                meta.get("name", "")
                or meta.get("title", "")
                or meta.get("source", "")
            )
            if doc_name:
                docs_info.add(doc_name)
        return ", ".join(docs_info)

    log.debug(
        f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}"
    )

    # ── Layer 3: Document-level exact dedup ──────────────────────────────────
    # If the same content (same hash) is already in this collection, skip the
    # entire pipeline — no chunking, no embedding, no vector store writes.
    # We do NOT distinguish by file_id here: re-uploading the same file under
    # a new file_id would otherwise create duplicate chunks. The hash is
    # computed from the file's full text content, so an identical hash in the
    # same collection means the content is already fully indexed.
    # (Different collections are naturally separated by collection_name.)
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]},
        )
        if result is not None and result.ids and len(result.ids) > 0:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                existing_file_id = None
                if result.metadatas and result.metadatas[0]:
                    existing_file_id = result.metadatas[0][0].get("file_id")

                new_file_id = metadata.get("file_id")
                if existing_file_id and existing_file_id != new_file_id:
                    # Re-upload of same content under a new file_id — still a
                    # duplicate in this collection; skip to prevent ghost chunks.
                    log.info(
                        "[DEDUP-L3] Same content already indexed under a different "
                        "file_id — skipping to prevent duplicates "
                        "(existing=%s, new=%s, hash=%s, collection=%s)",
                        existing_file_id, new_file_id,
                        metadata["hash"][:16], collection_name,
                    )
                else:
                    # Exact same file re-added
                    log.info(
                        "[DEDUP-L3] Exact duplicate document skipped — "
                        "hash=%s file_id=%s collection=%s",
                        metadata["hash"][:16],
                        existing_file_id or new_file_id,
                        collection_name,
                    )
                return True   # always skip when hash already in this collection

    # Capture full document text BEFORE splitting — context window for
    # contextual chunking (Anthropic paper approach).
    full_doc_text   = " ".join(doc.page_content for doc in docs)
    _pipeline_start = _time.monotonic()
    _raw_char_count = len(full_doc_text)
    print("━" * 62, flush=True)
    print(f"PIPELINE START  collection={collection_name}", flush=True)
    print(f"  Raw text : {_raw_char_count:,} chars  |  Input docs: {len(docs)}", flush=True)

    from futureverse_ai.retrieval.pipeline_agents import run_ingestion_pipeline
    try:
        return run_ingestion_pipeline(
            docs=docs,
            collection_name=collection_name,
            full_doc_text=full_doc_text,
            request=request,
            metadata=metadata,
            overwrite=overwrite,
            split=split,
            add=add,
            user=user,
            pipeline_start=_pipeline_start,
            merge_fn=merge_docs_to_target_size,
        )
    except Exception as e:
        log.exception(e)
        raise e


class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None


@router.post("/process/file")
def process_file(
    request: Request,
    form_data: ProcessFileForm,
    user=Depends(get_verified_user),
    db: Session = Depends(get_session),
):
    """
    Process a file and save its content to the vector database.
    Process a file and save its content to the vector database.
    Note: granular session management is used to prevent connection pool exhaustion.
    The session is committed before external API calls, and updates use a fresh session.
    """
    if user.role == "admin":
        file = Files.get_file_by_id(form_data.file_id, db=db)
    else:
        file = Files.get_file_by_id_and_user_id(form_data.file_id, user.id, db=db)

    if file:
        try:

            collection_name = form_data.collection_name

            if collection_name is None:
                collection_name = f"file-{file.id}"

            if form_data.content:
                # Update the content in the file
                # Usage: /files/{file_id}/data/content/update, /files/ (audio file upload pipeline)

                try:
                    # /files/{file_id}/data/content/update
                    VECTOR_DB_CLIENT.delete_collection(
                        collection_name=f"file-{file.id}"
                    )
                except:
                    # Audio file upload pipeline
                    pass

                docs = [
                    Document(
                        page_content=form_data.content.replace("<br/>", "\n"),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]

                text_content = form_data.content
            elif form_data.collection_name:
                # Check if the file has already been processed and save the content
                # Usage: /knowledge/{id}/file/add, /knowledge/{id}/file/update

                # Explicit high limit: without it the vector DB applies its
                # default query limit (Weaviate = 25), which silently truncates
                # large files to the first 25 chunks when added to a KB.
                result = VECTOR_DB_CLIENT.query(
                    collection_name=f"file-{file.id}",
                    filter={"file_id": file.id},
                    limit=10000,
                )

                if result is not None and len(result.ids[0]) > 0:
                    docs = [
                        Document(
                            page_content=result.documents[0][idx],
                            metadata=result.metadatas[0][idx],
                        )
                        for idx, id in enumerate(result.ids[0])
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]

                text_content = file.data.get("content", "")
            else:
                # Process the file and save the content
                # Usage: /files/
                file_path = file.path
                if file_path:
                    file_path = Storage.get_file(file_path)
                    loader = Loader(
                        engine=request.app.state.config.CONTENT_EXTRACTION_ENGINE,
                        user=user,
                        DATALAB_MARKER_API_KEY=request.app.state.config.DATALAB_MARKER_API_KEY,
                        DATALAB_MARKER_API_BASE_URL=request.app.state.config.DATALAB_MARKER_API_BASE_URL,
                        DATALAB_MARKER_ADDITIONAL_CONFIG=request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
                        DATALAB_MARKER_SKIP_CACHE=request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
                        DATALAB_MARKER_FORCE_OCR=request.app.state.config.DATALAB_MARKER_FORCE_OCR,
                        DATALAB_MARKER_PAGINATE=request.app.state.config.DATALAB_MARKER_PAGINATE,
                        DATALAB_MARKER_STRIP_EXISTING_OCR=request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
                        DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION=request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
                        DATALAB_MARKER_FORMAT_LINES=request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
                        DATALAB_MARKER_USE_LLM=request.app.state.config.DATALAB_MARKER_USE_LLM,
                        DATALAB_MARKER_OUTPUT_FORMAT=request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
                        EXTERNAL_DOCUMENT_LOADER_URL=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
                        EXTERNAL_DOCUMENT_LOADER_API_KEY=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
                        TIKA_SERVER_URL=request.app.state.config.TIKA_SERVER_URL,
                        DOCLING_SERVER_URL=request.app.state.config.DOCLING_SERVER_URL,
                        DOCLING_API_KEY=request.app.state.config.DOCLING_API_KEY,
                        DOCLING_PARAMS=request.app.state.config.DOCLING_PARAMS,
                        PDF_EXTRACT_IMAGES=request.app.state.config.PDF_EXTRACT_IMAGES,
                        DOCUMENT_INTELLIGENCE_ENDPOINT=request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
                        DOCUMENT_INTELLIGENCE_KEY=request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
                        DOCUMENT_INTELLIGENCE_MODEL=request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
                        MISTRAL_OCR_API_BASE_URL=request.app.state.config.MISTRAL_OCR_API_BASE_URL,
                        MISTRAL_OCR_API_KEY=request.app.state.config.MISTRAL_OCR_API_KEY,
                        MINERU_API_MODE=request.app.state.config.MINERU_API_MODE,
                        MINERU_API_URL=request.app.state.config.MINERU_API_URL,
                        MINERU_API_KEY=request.app.state.config.MINERU_API_KEY,
                        MINERU_API_TIMEOUT=request.app.state.config.MINERU_API_TIMEOUT,
                        MINERU_PARAMS=request.app.state.config.MINERU_PARAMS,
                    )
                    docs = loader.load(
                        file.filename, file.meta.get("content_type"), file_path
                    )

                    docs = [
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **filter_metadata(doc.metadata),
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                        for doc in docs
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]
                text_content = " ".join([doc.page_content for doc in docs])

            log.debug(f"text_content: {text_content}")
            Files.update_file_data_by_id(
                file.id,
                {"content": text_content},
                db=db,
            )
            hash = calculate_sha256_string(text_content)

            if request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:
                Files.update_file_data_by_id(file.id, {"status": "completed"}, db=db)
                Files.update_file_hash_by_id(file.id, hash, db=db)
                return {
                    "status": True,
                    "collection_name": None,
                    "filename": file.filename,
                    "content": text_content,
                }
            else:
                try:
                    # Commit any pending changes before the slow embedding step.
                    # Note: file is already a Pydantic model (not ORM), so no expunge needed.
                    db.commit()

                    # External embedding API takes time (5-60s+).
                    # Subsequent updates use fresh sessions via get_db().
                    result = save_docs_to_vector_db(
                        request,
                        docs=docs,
                        collection_name=collection_name,
                        metadata={
                            "file_id": file.id,
                            "name": file.filename,
                            "hash": hash,
                        },
                        add=(True if form_data.collection_name else False),
                        user=user,
                    )
                    log.info(f"added {len(docs)} items to collection {collection_name}")

                    if result:
                        # Fresh session for the final update.
                        with get_db() as session:
                            Files.update_file_metadata_by_id(
                                file.id,
                                {
                                    "collection_name": collection_name,
                                },
                                db=session,
                            )

                            Files.update_file_data_by_id(
                                file.id,
                                {"status": "completed"},
                                db=session,
                            )
                            Files.update_file_hash_by_id(file.id, hash, db=session)

                            return {
                                "status": True,
                                "collection_name": collection_name,
                                "filename": file.filename,
                                "content": text_content,
                            }
                    else:
                        raise Exception("Error saving document to vector database")
                except Exception as e:
                    raise e

        except Exception as e:
            log.exception(e)
            # Fresh session for error status update.
            with get_db() as session:
                Files.update_file_data_by_id(
                    file.id,
                    {"status": "failed"},
                    db=session,
                )
                # Clear the hash so the file can be re-uploaded after fixing the issue
                Files.update_file_hash_by_id(file.id, None, db=session)

            if "No pandoc was found" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )


class ProcessTextForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None


@router.post("/process/text")
async def process_text(
    request: Request,
    form_data: ProcessTextForm,
    user=Depends(get_verified_user),
):
    collection_name = form_data.collection_name
    if collection_name is None:
        collection_name = calculate_sha256_string(form_data.content)

    docs = [
        Document(
            page_content=form_data.content,
            metadata={"name": form_data.name, "created_by": user.id},
        )
    ]
    text_content = form_data.content
    log.debug(f"text_content: {text_content}")

    result = await run_in_threadpool(
        save_docs_to_vector_db, request, docs, collection_name, user=user
    )
    if result:
        return {
            "status": True,
            "collection_name": collection_name,
            "content": text_content,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


@router.post("/process/youtube")
@router.post("/process/web")
async def process_web(
    request: Request,
    form_data: ProcessUrlForm,
    process: bool = Query(True, description="Whether to process and save the content"),
    user=Depends(get_verified_user),
):
    try:
        content, docs = await run_in_threadpool(
            get_content_from_url, request, form_data.url
        )
        log.debug(f"text_content: {content}")

        if process:
            collection_name = form_data.collection_name
            if not collection_name:
                collection_name = calculate_sha256_string(form_data.url)[:63]

            if not request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=True,
                    user=user,
                )
            else:
                collection_name = None

            return {
                "status": True,
                "collection_name": collection_name,
                "filename": form_data.url,
                "file": {
                    "data": {
                        "content": content,
                    },
                    "meta": {
                        "name": form_data.url,
                        "source": form_data.url,
                    },
                },
            }
        else:
            return {
                "status": True,
                "content": content,
            }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def search_web(
    request: Request, engine: str, query: str, user=None
) -> list[SearchResult]:
    """Search the web using a search engine and return the results as a list of SearchResult objects.
    Will look for a search engine API key in environment variables in the following order:
    - SEARXNG_QUERY_URL
    - YACY_QUERY_URL + YACY_USERNAME + YACY_PASSWORD
    - GOOGLE_PSE_API_KEY + GOOGLE_PSE_ENGINE_ID
    - BRAVE_SEARCH_API_KEY
    - KAGI_SEARCH_API_KEY
    - MOJEEK_SEARCH_API_KEY
    - BOCHA_SEARCH_API_KEY
    - SERPSTACK_API_KEY
    - SERPER_API_KEY
    - SERPLY_API_KEY
    - TAVILY_API_KEY
    - EXA_API_KEY
    - PERPLEXITY_API_KEY
    - SOUGOU_API_SID + SOUGOU_API_SK
    - SEARCHAPI_API_KEY + SEARCHAPI_ENGINE (by default `google`)
    - SERPAPI_API_KEY + SERPAPI_ENGINE (by default `google`)
    Args:
        query (str): The query to search for
    """

    # TODO: add playwright to search the web
    if engine == "ollama_cloud":
        return search_ollama_cloud(
            "https://ollama.com",
            request.app.state.config.OLLAMA_CLOUD_WEB_SEARCH_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "perplexity_search":
        if request.app.state.config.PERPLEXITY_API_KEY:
            return search_perplexity_search(
                request.app.state.config.PERPLEXITY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                request.app.state.config.PERPLEXITY_SEARCH_API_URL,
                user,
            )
        else:
            raise Exception("No PERPLEXITY_API_KEY found in environment variables")
    elif engine == "searxng":
        if request.app.state.config.SEARXNG_QUERY_URL:
            searxng_kwargs = {"language": request.app.state.config.SEARXNG_LANGUAGE}
            return search_searxng(
                request.app.state.config.SEARXNG_QUERY_URL,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                **searxng_kwargs,
            )
        else:
            raise Exception("No SEARXNG_QUERY_URL found in environment variables")
    elif engine == "yacy":
        if request.app.state.config.YACY_QUERY_URL:
            return search_yacy(
                request.app.state.config.YACY_QUERY_URL,
                request.app.state.config.YACY_USERNAME,
                request.app.state.config.YACY_PASSWORD,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No YACY_QUERY_URL found in environment variables")
    elif engine == "google_pse":
        if (
            request.app.state.config.GOOGLE_PSE_API_KEY
            and request.app.state.config.GOOGLE_PSE_ENGINE_ID
        ):
            return search_google_pse(
                request.app.state.config.GOOGLE_PSE_API_KEY,
                request.app.state.config.GOOGLE_PSE_ENGINE_ID,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                referer=request.app.state.config.WEBUI_URL,
            )
        else:
            raise Exception(
                "No GOOGLE_PSE_API_KEY or GOOGLE_PSE_ENGINE_ID found in environment variables"
            )
    elif engine == "brave":
        if request.app.state.config.BRAVE_SEARCH_API_KEY:
            return search_brave(
                request.app.state.config.BRAVE_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No BRAVE_SEARCH_API_KEY found in environment variables")
    elif engine == "kagi":
        if request.app.state.config.KAGI_SEARCH_API_KEY:
            return search_kagi(
                request.app.state.config.KAGI_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No KAGI_SEARCH_API_KEY found in environment variables")
    elif engine == "mojeek":
        if request.app.state.config.MOJEEK_SEARCH_API_KEY:
            return search_mojeek(
                request.app.state.config.MOJEEK_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No MOJEEK_SEARCH_API_KEY found in environment variables")
    elif engine == "bocha":
        if request.app.state.config.BOCHA_SEARCH_API_KEY:
            return search_bocha(
                request.app.state.config.BOCHA_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No BOCHA_SEARCH_API_KEY found in environment variables")
    elif engine == "serpstack":
        if request.app.state.config.SERPSTACK_API_KEY:
            return search_serpstack(
                request.app.state.config.SERPSTACK_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                https_enabled=request.app.state.config.SERPSTACK_HTTPS,
            )
        else:
            raise Exception("No SERPSTACK_API_KEY found in environment variables")
    elif engine == "serper":
        if request.app.state.config.SERPER_API_KEY:
            return search_serper(
                request.app.state.config.SERPER_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPER_API_KEY found in environment variables")
    elif engine == "serply":
        if request.app.state.config.SERPLY_API_KEY:
            return search_serply(
                request.app.state.config.SERPLY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                filter_list=request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPLY_API_KEY found in environment variables")
    elif engine == "duckduckgo":
        return search_duckduckgo(
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            concurrent_requests=request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            backend=request.app.state.config.DDGS_BACKEND,
        )
    elif engine == "tavily":
        if request.app.state.config.TAVILY_API_KEY:
            return search_tavily(
                request.app.state.config.TAVILY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No TAVILY_API_KEY found in environment variables")
    elif engine == "exa":
        if request.app.state.config.EXA_API_KEY:
            return search_exa(
                request.app.state.config.EXA_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No EXA_API_KEY found in environment variables")
    elif engine == "searchapi":
        if request.app.state.config.SEARCHAPI_API_KEY:
            return search_searchapi(
                request.app.state.config.SEARCHAPI_API_KEY,
                request.app.state.config.SEARCHAPI_ENGINE,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SEARCHAPI_API_KEY found in environment variables")
    elif engine == "serpapi":
        if request.app.state.config.SERPAPI_API_KEY:
            return search_serpapi(
                request.app.state.config.SERPAPI_API_KEY,
                request.app.state.config.SERPAPI_ENGINE,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPAPI_API_KEY found in environment variables")
    elif engine == "jina":
        return search_jina(
            request.app.state.config.JINA_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.JINA_API_BASE_URL,
        )
    elif engine == "bing":
        return search_bing(
            request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            str(DEFAULT_LOCALE),
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "azure":
        if (
            request.app.state.config.AZURE_AI_SEARCH_API_KEY
            and request.app.state.config.AZURE_AI_SEARCH_ENDPOINT
            and request.app.state.config.AZURE_AI_SEARCH_INDEX_NAME
        ):
            return search_azure(
                request.app.state.config.AZURE_AI_SEARCH_API_KEY,
                request.app.state.config.AZURE_AI_SEARCH_ENDPOINT,
                request.app.state.config.AZURE_AI_SEARCH_INDEX_NAME,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception(
                "AZURE_AI_SEARCH_API_KEY, AZURE_AI_SEARCH_ENDPOINT, and AZURE_AI_SEARCH_INDEX_NAME are required for Azure AI Search"
            )
    elif engine == "exa":
        return search_exa(
            request.app.state.config.EXA_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "perplexity":
        return search_perplexity(
            request.app.state.config.PERPLEXITY_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            model=request.app.state.config.PERPLEXITY_MODEL,
            search_context_usage=request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
        )
    elif engine == "sougou":
        if (
            request.app.state.config.SOUGOU_API_SID
            and request.app.state.config.SOUGOU_API_SK
        ):
            return search_sougou(
                request.app.state.config.SOUGOU_API_SID,
                request.app.state.config.SOUGOU_API_SK,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception(
                "No SOUGOU_API_SID or SOUGOU_API_SK found in environment variables"
            )
    elif engine == "firecrawl":
        return search_firecrawl(
            request.app.state.config.FIRECRAWL_API_BASE_URL,
            request.app.state.config.FIRECRAWL_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "external":
        return search_external(
            request,
            request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            user=user,
        )
    elif engine == "yandex":
        return search_yandex(
            request,
            request.app.state.config.YANDEX_WEB_SEARCH_URL,
            request.app.state.config.YANDEX_WEB_SEARCH_API_KEY,
            request.app.state.config.YANDEX_WEB_SEARCH_CONFIG,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            user=user,
        )
    else:
        raise Exception("No search engine API key found in environment variables")


@router.post("/process/web/search")
async def process_web_search(
    request: Request, form_data: SearchForm, user=Depends(get_verified_user)
):
    if not request.app.state.config.ENABLE_WEB_SEARCH:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    if user.role != "admin" and not has_permission(
        user.id, "features.web_search", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    urls = []
    result_items = []

    try:
        logging.debug(
            f"trying to web search with {request.app.state.config.WEB_SEARCH_ENGINE, form_data.queries}"
        )

        # Use semaphore to limit concurrent requests based on WEB_SEARCH_CONCURRENT_REQUESTS
        # 0 or None = unlimited (previous behavior), positive number = limited concurrency
        # Set to 1 for sequential execution (rate-limited APIs like Brave free tier)
        concurrent_limit = request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS

        if concurrent_limit:
            # Limited concurrency with semaphore
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def search_with_limit(query):
                async with semaphore:
                    return await run_in_threadpool(
                        search_web,
                        request,
                        request.app.state.config.WEB_SEARCH_ENGINE,
                        query,
                        user,
                    )

            search_tasks = [search_with_limit(query) for query in form_data.queries]
        else:
            # Unlimited parallel execution (previous behavior)
            search_tasks = [
                run_in_threadpool(
                    search_web,
                    request,
                    request.app.state.config.WEB_SEARCH_ENGINE,
                    query,
                    user,
                )
                for query in form_data.queries
            ]

        search_results = await asyncio.gather(*search_tasks)

        for result in search_results:
            if result:
                for item in result:
                    if item and item.link:
                        result_items.append(item)
                        urls.append(item.link)

        urls = list(dict.fromkeys(urls))
        log.debug(f"urls: {urls}")

    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.WEB_SEARCH_ERROR(e),
        )

    if len(urls) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.DEFAULT("No results found from web search"),
        )

    try:
        if request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER:
            search_results = [
                item for result in search_results for item in result if result
            ]

            docs = [
                Document(
                    page_content=result.snippet,
                    metadata={
                        "source": result.link,
                        "title": result.title,
                        "snippet": result.snippet,
                        "link": result.link,
                    },
                )
                for result in search_results
                if hasattr(result, "snippet") and result.snippet is not None
            ]
        else:
            loader = get_web_loader(
                urls,
                verify_ssl=request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
                requests_per_second=request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
                trust_env=request.app.state.config.WEB_SEARCH_TRUST_ENV,
            )
            docs = await loader.aload()

        urls = [
            doc.metadata.get("source") for doc in docs if doc.metadata.get("source")
        ]  # only keep the urls returned by the loader
        result_items = [
            dict(item) for item in result_items if item.link in urls
        ]  # only keep the search results that have been loaded

        if request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
            return {
                "status": True,
                "collection_name": None,
                "filenames": urls,
                "items": result_items,
                "docs": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in docs
                ],
                "loaded_count": len(docs),
            }
        else:
            # Create a single collection for all documents
            collection_name = (
                f"web-search-{calculate_sha256_string('-'.join(form_data.queries))}"[
                    :63
                ]
            )

            try:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=True,
                    user=user,
                )
            except Exception as e:
                log.debug(f"error saving docs: {e}")

            return {
                "status": True,
                "collection_names": [collection_name],
                "items": result_items,
                "filenames": urls,
                "loaded_count": len(docs),
            }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    k_reranker: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@router.post("/query/doc")
async def query_doc_handler(
    request: Request,
    form_data: QueryDocForm,
    user=Depends(get_verified_user),
):
    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (
            form_data.hybrid is None or form_data.hybrid
        ):
            collection_results = {}
            collection_results[form_data.collection_name] = VECTOR_DB_CLIENT.peek(
                collection_name=form_data.collection_name, limit=10
            )
            return await query_doc_with_hybrid_search(
                collection_name=form_data.collection_name,
                collection_sample=collection_results[form_data.collection_name],
                query=form_data.query,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (
                        lambda query, documents: request.app.state.RERANKING_FUNCTION(
                            query, documents, user=user
                        )
                    )
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker
                or request.app.state.config.TOP_K_RERANKER,
                r=(
                    form_data.r
                    if form_data.r
                    else request.app.state.config.RELEVANCE_THRESHOLD
                ),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
                user=user,
            )
        else:
            query_embedding = await request.app.state.EMBEDDING_FUNCTION(
                form_data.query, prefix=RAG_EMBEDDING_QUERY_PREFIX, user=user
            )
            return query_doc(
                collection_name=form_data.collection_name,
                query_embedding=query_embedding,
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                user=user,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryCollectionsForm(BaseModel):
    collection_names: list[str]
    query: str
    k: Optional[int] = None
    k_reranker: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None
    hybrid_bm25_weight: Optional[float] = None
    enable_enriched_texts: Optional[bool] = None


@router.post("/query/collection")
async def query_collection_handler(
    request: Request,
    form_data: QueryCollectionsForm,
    user=Depends(get_verified_user),
):
    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (
            form_data.hybrid is None or form_data.hybrid
        ):
            return await query_collection_with_hybrid_search(
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (
                        lambda query, documents: request.app.state.RERANKING_FUNCTION(
                            query, documents, user=user
                        )
                    )
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker
                or request.app.state.config.TOP_K_RERANKER,
                r=(
                    form_data.r
                    if form_data.r
                    else request.app.state.config.RELEVANCE_THRESHOLD
                ),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
                enable_enriched_texts=(
                    form_data.enable_enriched_texts
                    if form_data.enable_enriched_texts is not None
                    else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
                ),
            )
        else:
            return await query_collection(
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


####################################
#
# Vector DB operations
#
####################################


class DeleteForm(BaseModel):
    collection_name: str
    file_id: str


@router.post("/delete")
def delete_entries_from_collection(
    form_data: DeleteForm,
    user=Depends(get_admin_user),
    db: Session = Depends(get_session),
):
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=form_data.collection_name):
            file = Files.get_file_by_id(form_data.file_id, db=db)
            if not file:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND,
                )
            hash = file.hash

            VECTOR_DB_CLIENT.delete(
                collection_name=form_data.collection_name,
                metadata={"hash": hash},
            )
            return {"status": True}
        else:
            return {"status": False}
    except Exception as e:
        log.exception(e)
        return {"status": False}


@router.post("/reset/db")
def reset_vector_db(user=Depends(get_admin_user), db: Session = Depends(get_session)):
    VECTOR_DB_CLIENT.reset()
    Knowledges.delete_all_knowledge(db=db)


@router.post("/reset/uploads")
def reset_upload_dir(user=Depends(get_admin_user)) -> bool:
    folder = f"{UPLOAD_DIR}"
    try:
        # Check if the directory exists
        if os.path.exists(folder):
            # Iterate over all the files and directories in the specified directory
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    log.exception(f"Failed to delete {file_path}. Reason: {e}")
        else:
            log.warning(f"The directory {folder} does not exist")
    except Exception as e:
        log.exception(f"Failed to process the directory {folder}. Reason: {e}")
    return True


if ENV == "dev":

    @router.get("/ef/{text}")
    async def get_embeddings(request: Request, text: Optional[str] = "Hello World!"):
        return {
            "result": await request.app.state.EMBEDDING_FUNCTION(
                text, prefix=RAG_EMBEDDING_QUERY_PREFIX
            )
        }


class BatchProcessFilesForm(BaseModel):
    files: List[FileModel]
    collection_name: str


class BatchProcessFilesResult(BaseModel):
    file_id: str
    status: str
    error: Optional[str] = None


class BatchProcessFilesResponse(BaseModel):
    results: List[BatchProcessFilesResult]
    errors: List[BatchProcessFilesResult]


@router.post("/process/files/batch")
async def process_files_batch(
    request: Request,
    form_data: BatchProcessFilesForm,
    user=Depends(get_verified_user),
) -> BatchProcessFilesResponse:
    """
    Process a batch of files and save them to the vector database.

    NOTE: We intentionally do NOT use Depends(get_session) here.
    The save_docs_to_vector_db() call makes external embedding API calls which
    can take 5-60+ seconds for batch operations. Database operations after
    embedding (Files.update_file_by_id) manage their own short-lived sessions.
    """

    collection_name = form_data.collection_name

    file_results: List[BatchProcessFilesResult] = []
    file_errors: List[BatchProcessFilesResult] = []
    file_updates: List[FileUpdateForm] = []

    # Prepare all documents first
    all_docs: List[Document] = []

    for file in form_data.files:
        try:
            text_content = file.data.get("content", "")
            file_hash = calculate_sha256_string(text_content)
            docs: List[Document] = [
                Document(
                    page_content=text_content.replace("<br/>", "\n"),
                    metadata={
                        **file.meta,
                        "name": file.filename,
                        "created_by": file.user_id,
                        "file_id": file.id,
                        "source": file.filename,
                        # Include the file-level hash so ChunkDeduplicationAgent
                        # can detect re-uploaded files (same content, new file_id).
                        "hash": file_hash,
                    },
                )
            ]

            all_docs.extend(docs)

            file_updates.append(
                FileUpdateForm(
                    hash=file_hash,
                    data={"content": text_content},
                )
            )
            file_results.append(
                BatchProcessFilesResult(file_id=file.id, status="prepared")
            )

        except Exception as e:
            log.error(f"process_files_batch: Error processing file {file.id}: {str(e)}")
            file_errors.append(
                BatchProcessFilesResult(file_id=file.id, status="failed", error=str(e))
            )

    # Save all documents in one batch
    if all_docs:
        try:
            await run_in_threadpool(
                save_docs_to_vector_db,
                request,
                all_docs,
                collection_name,
                add=True,
                user=user,
            )

            # Update all files with collection name
            for file_update, file_result in zip(file_updates, file_results):
                Files.update_file_by_id(id=file_result.file_id, form_data=file_update)
                file_result.status = "completed"

        except Exception as e:
            log.error(
                f"process_files_batch: Error saving documents to vector DB: {str(e)}"
            )
            for file_result in file_results:
                file_result.status = "failed"
                file_errors.append(
                    BatchProcessFilesResult(
                        file_id=file_result.file_id, status="failed", error=str(e)
                    )
                )

    return BatchProcessFilesResponse(results=file_results, errors=file_errors)

