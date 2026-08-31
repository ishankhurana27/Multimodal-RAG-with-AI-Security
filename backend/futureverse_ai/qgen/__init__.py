"""Serving layer for the fine-tuned semantic question-generation model.

This package is the backend twin of the qgen training pipeline:

  render.py          verbatim copy of qgen/scripts/render.py — the deterministic
                     renderer that owns everything letter-shaped (answer codes,
                     shuffle, letters, key). Sync it whenever the training copy
                     changes; it self-tests with `python render.py`.
  system_prompt.txt  byte-identical to qgen/data/sft/system_prompt.txt (written
                     by 06_build_sft.py) and to the SYSTEM string baked into the
                     Ollama Modelfile. If these ever differ, the fine-tune is
                     being served a prompt it never saw.
  engine.py          prompt builders (byte-compatible with 06_build_sft.py's
                     control_block / evidence_block), the direct Ollama client
                     in the exact request shape evaluate.py validated, and the
                     serving guardrails recommended by the four-arm eval verdict
                     (reports/verdict_qwen3_sft.md).
"""
