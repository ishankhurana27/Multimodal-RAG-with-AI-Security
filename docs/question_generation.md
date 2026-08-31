# UPSC Question Generation — setup and usage

Dual-retrieval RAG question generation: content chunks come from a knowledge
base (factual grounding), previous-year questions come from a dedicated
`PyqKnowledge` collection (style exemplars). The two channels are never merged
— merging makes retrieval pull PYQs when you wanted content, and the model
starts reproducing past papers.

## 1. Build the generation model

    ollama create upsc-qgen -f docs/upsc-qgen.Modelfile
    ollama run upsc-qgen "say ok"   # sanity check

The Modelfile sets `num_ctx 16384` — without it Ollama's 2048 default silently
truncates the RAG context.

## 2. Recommended retrieval settings (Admin Panel → Settings → Documents)

- Embedding model: `BAAI/bge-m3` (or keep `bge-large-en-v1.5` if re-embedding
  everything is too costly right now — do not mix models within a collection)
- Hybrid search: ON (UPSC content is full of exact-match terms — Article 356,
  PESA Act — where pure semantic search underperforms)
- Reranker: `BAAI/bge-reranker-v2-m3`
- Chunk size 900–1200 tokens, overlap 150 for books
- PYQs are ingested one question per chunk automatically (never split)

## 3. Normalize and ingest PYQs

Prepare a CSV/XLSX with columns: `question, year, paper, subject, topic,
subtopic, marks, word_limit, command_word, type` (extra columns ignored;
`options`/`answer` for MCQs). Then:

    # Normalize only — writes pyq_normalized.jsonl (keep for analytics + LoRA later)
    python scripts/pyq_normalize.py mains_2013_2024.xlsx

    # Normalize + ingest into the backend (admin API key required)
    python scripts/pyq_normalize.py mains_2013_2024.xlsx \
        --ingest --url http://localhost:8080 --api-key sk-xxxx

Fill in `subject` and `topic` for every record — exemplar retrieval filters on
them. The script warns about records missing these.

Check what went in:

    GET /api/v1/questions/pyq/stats

Returns topic frequency and command-word distribution — use it to weight which
topics you generate for (UPSC skews heavily to Discuss / Examine / Critically
examine / Elucidate).

## 4. Generate questions

    POST /api/v1/questions/generate
    {
      "model": "upsc-qgen",
      "knowledge_id": "<knowledge base id>",
      "subject": "Polity",
      "topic": "Federalism",
      "paper": "GS-II",
      "question_type": "descriptive",   // or "mcq"
      "count": 5,
      "marks": 15,
      "word_limit": 250
    }

Response contains the generated questions with, per question:

- `source_chunk_ref` — which SOURCE chunk the facts came from (hallucination
  brake: makes review fast)
- `pyq_similarity` + `near_duplicate` — max cosine similarity against the PYQ
  collection; anything ≥ 0.88 is flagged as a near-duplicate of a past paper

## 5. Review workflow (do not skip)

A 7B model produces structurally correct questions but WILL hallucinate
specifics (Article numbers, committee names, scheme dates). Treat every
generated question as requiring human factual verification against its
`source_chunk_ref`. Reject:

- anything flagged `near_duplicate`
- any factual referent not present in the source chunk
- anything answerable in one sentence (recall, not analysis)

## 6. When to fine-tune (phase 3, not phase 1)

Only after few-shot generation works and you see a persistent stylistic gap:
Unsloth → LoRA r=16 alpha=32 on ~2,500 PYQ instruction pairs (from
`pyq_normalized.jsonl`) → merge → convert to GGUF → `FROM ./model.gguf` in the
Modelfile. It improves phrasing and command-word idiom; it will NOT fix
factual hallucination — that is what RAG is for.
