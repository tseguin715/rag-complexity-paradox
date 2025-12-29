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

- `run_baseline.py` — baseline + hybrid + agentic evaluation harness (Systems A–C)  
- `run_godmode.py` — “God Mode” wide-retrieval + reranking harness (System D)  
- `model_factory.py` — model/provider configuration (OpenAI, Gemini, Ollama local models)  
- `results_table.csv` — aggregated results (one row per configuration)  
- `evaluation_results_*.jsonl` — per-question logs (one row per question per config)
- `create_source_data.py` - For creating the raw database file `source_data.ndjson` via TMDB's API
- `tmdb_ids_combined.csv` - A dependency for `create_source_data.py` - containing the TMDB IDs of the TV/Movie titles
- `gold_set_update.py` - For creating a time-accurate 'gold set' for the RAG evaluation harness
- `gold_set_template.jsonl` - A dependency for `gold_set_update.py` - containing the 'base' gold set that will be adjusted
- `build_child_parent_db.py` - For creating the four vector databases

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

## Setup

### 1) Set up environment (Python 3.10)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install PyTorch (CUDA) + requirements
Install your CUDA-enabled PyTorch build first. If you're only on CPU, install the CPU PyTorch build instead of CUDA.

then:

```bash
pip install -r requirements.txt
```

### 3) Set API keys (for hosted models + judging)

This repo uses hosted APIs for some configs (e.g., GPT judge, OpenAI/Gemini LLMs). Set whichever you plan to run:

macOS/Linux:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "..."
setx GOOGLE_API_KEY "..."
```

### 4) Local LLMs (optional, via Ollama)

If you run local configs like qwen2.5:32b / llama3.1:70b, install Ollama and pull the models:

```bash
ollama pull qwen2.5:32b
ollama pull qwen2.5:32b-instruct-q2_K
ollama pull llama3.1:70b
ollama pull bazsalanszky/llama3.1:70B-instruct-iq2_x
```

### 5) First run note (Hugging Face downloads)

Embedding models (e.g., Jina / BGE) may download weights the first time you run them and cache locally. 

## Data

This repo does not include TMDB-derived datasets or prebuilt Chroma DBs (to avoid redistributing TMDB content).  
To reproduce results, you must obtain TMDB data and create the vector databases yourself. There is also a requirement to make a 'gold_set.jsonl' that is accurate for the present time.
Scripts are provided to do these steps.

### 1) Create source_data.ndjson

Obtain an API key from tmdb.org and run:

```bash
python create_source_data.py --api_key 'YOUR_KEY_HERE'
```

This will create source_data.ndjson. This process requires tmdb_ids_combined.csv, which contains the databases's movie/TV IDs. This process will take several hours -- it took me ~3 hours during a test run.

### 2) Create gold_set.jsonl

This will create a 'gold_set.jsonl' for the evaluation harness that contains gold answers that are current for the time the data was extracted.
This step needs to be done because some values (e.g. ratings and popularity) drift over time.

For example, the answers for these questions were different at the time the experiments were actually performed and when the data was re-extracted for a test run:

question_id | question | 2025.11.12 | 2025.12.27
-------- | -------- | -------- | -------- 
Q_035 | What is the rating of the movie Forrest Gump (NOT "Through the Eyes of Forrest Gump")? | 8.466 | 8.5
Q_097 | How many votes did "My Dog Skip" receive? | 279 | 284


The gold_set.json creation step depends on gold_set_template.jsonl and source_data.ndjson. It also creates a drift_report.csv that shows the answers that have changed.

```bash
python gold_set_update.py
```

### 3) Create the vector databases

This will create the four databases in the db subdirectory using source_data.ndjson:

```bash
python build_child_parent_db.py
```





