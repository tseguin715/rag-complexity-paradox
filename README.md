# The RAG Complexity Paradox  
**Why 20× more spend didn’t beat a ~70% ceiling**

This repo contains the full code + outputs for an empirical RAG evaluation: **96 configurations** across **4 architectures**, **6 LLMs**, and **4 embedding models** on a **100-question TMDB QA benchmark**.

**Headline results**
- **Complexity ≠ accuracy:** the best “God Mode” setup reached **~71.7% correctness**, while a much simpler baseline was **~70%** (not statistically separable in this benchmark).
- **There’s a clear sweet spot:** **moderate agentic control** (query rewriting + doc grading) sits on the best accuracy/latency frontier.
- **Complex pipelines can be fragile:** wide retrieval + reranking introduces failure modes (latency blowups, context overload).
- **Quantization fragility is model-dependent:** **4→2-bit** compression can be catastrophic for some models inside complex pipelines (e.g., Llama IQ2), while others degrade modestly.

📄 **Read the write-up:** *(link to your hosted article)*  
🧪 **Reproducible results:** JSONL logs + aggregate tables are produced by the harness scripts below.

---

## What’s in here

- `main_cli_fixed_db.py` — baseline + hybrid + agentic evaluation harness (Systems A–C)  
- `iter_reranked_fixed_fixed.py` — “God Mode” wide-retrieval + reranking harness (System D)  
- `model_factory.py` — model/provider configuration (OpenAI, Gemini, Ollama local models)  
- `results_table.csv` — aggregated results (one row per configuration)  
- `evaluation_results_*.jsonl` — per-question logs (one row per question per config)

**Metrics**
- `correctness_score` / `faithfulness_score`: LLM-as-judge (fixed judge, temp=0) over (question, gold answer, model answer)
- Retrieval metrics: recall/MRR against gold document IDs (see article for definitions)

---

## Systems evaluated (high level)

- **System A (Naive):** vector search (k=20) → stuff context → generate  
- **System B (Hybrid):** vector + metadata filtering + specialized handling  
- **System C (Agentic):** query rewriting + doc grading + adaptive retrieval  
- **System D (God Mode):** wide retrieval (k=100) + cross-encoder reranking → top-k → generate

Full diagrams + details: see the article.

---

## Quickstart

### 1) Set up environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
