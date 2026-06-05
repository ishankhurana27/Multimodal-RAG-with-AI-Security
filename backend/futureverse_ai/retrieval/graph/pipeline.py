"""
pipeline.py — UPSC RAG Pipeline using LangGraph
Hybrid: Vector (Weaviate) + Graph (Neo4j) + BM25
"""
import os
import logging
import requests
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

try:
    from config import RAG_ANSWER_MODEL, OLLAMA_BASE_URL
except ImportError:
    RAG_ANSWER_MODEL = os.environ.get("RAG_ANSWER_MODEL", "qwen2.5:32b")
    OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

from futureverse_ai.retrieval.token_logger import (
    log_call, log_query_summary, new_query_id,
)

log = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

class UPSCRAGState(TypedDict):
    query:            str
    subject_filter:   Optional[str]      # Geography / History / Polity etc.
    exam_mode:        Optional[str]      # prelims / mains / both
    student_level:    Optional[int]      # 1-10 from student profile
    vector_results:   List[str]
    graph_results:    List[str]
    bm25_results:     List[str]
    combined_context: str
    answer:           str
    sources:          List[str]
    memory:           List[dict]
    collection_names: List[str]
    # ── Token tracking ────────────────────────────────────────────────────────
    query_id:         str            # unique ID shared with query_expansion stage
    token_usage:      dict           # populated by reasoning_node

# ── Nodes ─────────────────────────────────────────────────────────────────────

def vector_search_node(state: UPSCRAGState) -> UPSCRAGState:
    try:
        from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT
        from futureverse_ai.retrieval.utils import get_embedding_function

        ef = get_embedding_function(
            engine="ollama",
            model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
            url=OLLAMA_BASE_URL,
        )
        query_vector = ef([state["query"]])[0]

        results = []
        sources = []
        for collection_name in state.get("collection_names", []):
            try:
                search_result = VECTOR_DB_CLIENT.search(
                    collection_name=collection_name,
                    vectors=[query_vector],
                    limit=7,
                )
                if search_result and search_result.documents:
                    for doc_list in search_result.documents:
                        results.extend(doc_list)
                if search_result and search_result.metadatas:
                    for meta_list in search_result.metadatas:
                        for m in meta_list:
                            if isinstance(m, dict):
                                src = m.get("source") or m.get("name", "")
                                if src:
                                    sources.append(src)
            except Exception as e:
                log.warning(f"Vector search error for {collection_name}: {e}")

        return {**state, "vector_results": results, "sources": sources}
    except Exception as e:
        log.warning(f"Vector search failed: {e}")
        return {**state, "vector_results": [], "sources": []}


def graph_search_node(state: UPSCRAGState) -> UPSCRAGState:
    try:
        from extractor import search_graph

        # Use importance filter based on exam mode
        importance = None
        if state.get("exam_mode") == "prelims":
            importance = "high"

        results = search_graph(
            query=state["query"],
            limit=20,
            subject_filter=state.get("subject_filter"),
            importance_filter=importance,
            hops=2
        )
        return {**state, "graph_results": results}
    except Exception as e:
        log.warning(f"Graph search failed: {e}")
        return {**state, "graph_results": []}


def bm25_search_node(state: UPSCRAGState) -> UPSCRAGState:
    """BM25 keyword search over chunks."""
    try:
        import json
        from rank_bm25 import BM25Okapi

        chunks_dir = os.environ.get("CHUNKS_DIR", "../output_json/chunks")
        all_chunks = []

        # Load chunks for relevant subject only
        subject = (state.get("subject_filter") or "").lower()
        for fname in os.listdir(chunks_dir):
            if not fname.endswith(".json"):
                continue
            if subject and subject not in fname.lower():
                continue
            with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))

        if not all_chunks:
            return {**state, "bm25_results": []}

        tokenized = [c["text"].lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(state["query"].lower().split())
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:5]
        results = [all_chunks[i]["text"] for i in top_idx if scores[i] > 0]
        return {**state, "bm25_results": results}

    except Exception as e:
        log.warning(f"BM25 search failed: {e}")
        return {**state, "bm25_results": []}


def combine_context_node(state: UPSCRAGState) -> UPSCRAGState:
    parts = []

    if state["vector_results"]:
        parts.append(
            "=== DOCUMENT CHUNKS (Semantic Search) ===\n"
            + "\n---\n".join(state["vector_results"][:6])
        )

    if state["bm25_results"]:
        parts.append(
            "=== KEYWORD MATCHES (BM25) ===\n"
            + "\n---\n".join(state["bm25_results"][:4])
        )

    if state["graph_results"]:
        parts.append(
            "=== KNOWLEDGE GRAPH (Entity Relationships) ===\n"
            + "\n".join(state["graph_results"][:25])
        )

    combined = "\n\n".join(parts) if parts else "No context found."
    return {**state, "combined_context": combined}


def reasoning_node(state: UPSCRAGState) -> UPSCRAGState:
    # Adjust depth based on student level
    level = state.get("student_level", 5)
    depth_instruction = (
        "Use simple language suitable for a beginner."
        if level <= 3 else
        "Provide detailed UPSC-level analysis with examples."
        if level >= 7 else
        "Provide clear explanation suitable for intermediate UPSC preparation."
    )

    # Exam mode instruction
    exam_mode = state.get("exam_mode", "both")
    if exam_mode == "prelims":
        exam_instruction = "Focus on factual, MCQ-friendly information. Highlight key facts, dates, and named entities."
    elif exam_mode == "mains":
        exam_instruction = "Provide analytical depth suitable for 150-250 word Mains answers. Include multiple dimensions."
    else:
        exam_instruction = "Cover both factual accuracy (Prelims) and analytical depth (Mains)."

    # Memory context
    memory_ctx = ""
    if state.get("memory"):
        memory_ctx = "\n=== PREVIOUS CONVERSATION ===\n"
        for m in state["memory"][-3:]:
            memory_ctx += f"Q: {m['query']}\nA: {m['answer'][:300]}...\n\n"

    # Sources
    source_list = "\n".join(f"- {s}" for s in state.get("sources", [])[:5])

    prompt = f"""You are an expert UPSC Geography teacher and knowledge assistant.
Your answers must be:
1. Strictly based on the provided context
2. At UPSC examination standard
3. {depth_instruction}
4. {exam_instruction}

If the answer is not in the context, say: "This information is not available in the loaded study material."
Never make up facts. Always cite which book/chapter the information comes from.

=== STUDY MATERIAL CONTEXT ===
{state["combined_context"]}
{memory_ctx}
=== SOURCES AVAILABLE ===
{source_list}

=== STUDENT QUESTION ===
{state["query"]}

=== ANSWER ==="""

    # ── Direct Ollama call — gives us prompt_eval_count / eval_count ──────────
    # LangChain's OllamaLLM.invoke() does not reliably expose token counts,
    # so we call the Ollama API directly here.
    answer        = ""
    input_tokens  = 0
    output_tokens = 0

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":   RAG_ANSWER_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data          = resp.json()
        answer        = data.get("response", "").strip()
        input_tokens  = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count",        0)
    except Exception as exc:
        log.error(f"reasoning_node Ollama call failed: {exc}")
        answer = "Error: could not reach the language model. Please try again."

    # ── Token logging ─────────────────────────────────────────────────────────
    query_id = state.get("query_id") or new_query_id()
    query    = state["query"]

    call_record = log_call(
        query_id=query_id,
        stage="answer_generation",
        model=RAG_ANSWER_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        query_text=query,
    )

    # Pull query-expansion tokens that were logged earlier under the same query_id.
    # They live in state["token_usage"] if the caller threaded the query_id through
    # the expansion stage; fall back to 0 if expansion was not tracked this session.
    prev_usage    = state.get("token_usage") or {}
    exp_in        = prev_usage.get("query_expansion_input",  0)
    exp_out       = prev_usage.get("query_expansion_output", 0)
    exp_model     = prev_usage.get("expansion_model", "")

    summary = log_query_summary(
        query_id=query_id,
        query_text=query,
        query_expansion_in=exp_in,
        query_expansion_out=exp_out,
        expansion_model=exp_model,
        answer_in=input_tokens,
        answer_out=output_tokens,
        answer_model=RAG_ANSWER_MODEL,
    )

    token_usage = {
        "query_id":               query_id,
        "query_expansion_input":  exp_in,
        "query_expansion_output": exp_out,
        "query_expansion_total":  exp_in + exp_out,
        "answer_input":           input_tokens,
        "answer_output":          output_tokens,
        "answer_total":           input_tokens + output_tokens,
        "grand_total_tokens":     summary["grand_total_tokens"],
        "estimated_cost_usd":     summary["estimated_cost_usd"],
        "expansion_model":        exp_model,
        "answer_model":           RAG_ANSWER_MODEL,
    }

    memory = list(state.get("memory", []))
    memory.append({"query": query, "answer": answer})

    return {**state, "answer": answer, "memory": memory,
            "query_id": query_id, "token_usage": token_usage}

# ── Pipeline Build ─────────────────────────────────────────────────────────────

def build_pipeline():
    workflow = StateGraph(UPSCRAGState)

    workflow.add_node("vector_search",  vector_search_node)
    workflow.add_node("graph_search",   graph_search_node)
    workflow.add_node("bm25_search",    bm25_search_node)
    workflow.add_node("combine_context", combine_context_node)
    workflow.add_node("reasoning",      reasoning_node)

    workflow.set_entry_point("vector_search")

    # Vector and BM25 run in sequence, graph in parallel conceptually
    workflow.add_edge("vector_search",   "graph_search")
    workflow.add_edge("graph_search",    "bm25_search")
    workflow.add_edge("bm25_search",     "combine_context")
    workflow.add_edge("combine_context", "reasoning")
    workflow.add_edge("reasoning",       END)

    return workflow.compile()


_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_query(
    query: str,
    collection_names: List[str] = [],
    memory: List[dict] = [],
    subject_filter: str = None,
    exam_mode: str = "both",
    student_level: int = 5,
    query_id: str = None,
    query_expansion_token_usage: dict = None,
) -> dict:
    """
    Run the full UPSC RAG pipeline.

    Args:
        query_id:
            Optional ID from ``new_query_id()``.  Pass the same ID that was
            used when calling ``expand_queries_with_llm()`` / ``multi_query_retrieve()``
            so that the query-expansion token log and the answer-generation log
            are linked under one ``query_summary`` record.
            If omitted, a new ID is generated for this call only.
        query_expansion_token_usage:
            Optional dict with keys ``query_expansion_input``, ``query_expansion_output``,
            ``expansion_model`` filled in by the expansion stage so the summary
            record reflects both stages.  If omitted the summary shows 0 for expansion.
    """
    pipeline = get_pipeline()
    return pipeline.invoke({
        "query":            query,
        "subject_filter":   subject_filter,
        "exam_mode":        exam_mode,
        "student_level":    student_level,
        "vector_results":   [],
        "graph_results":    [],
        "bm25_results":     [],
        "combined_context": "",
        "answer":           "",
        "sources":          [],
        "memory":           memory,
        "collection_names": collection_names,
        "query_id":         query_id or new_query_id(),
        "token_usage":      query_expansion_token_usage or {},
    })