# The RAG Complexity Paradox  
**Why 20× more spend didn’t beat a ~70% ceiling**

This repo contains the full code + outputs for an empirical RAG evaluation: **96 configurations** across **4 architectures**, **6 LLMs**, and **4 embedding models** on a **100-question TMDB QA benchmark**.

**Headline results**
- **Complexity ≠ accuracy:** Top configs cluster around ~70% correctness; no consistent gain from "God Mode" (highest complexity architecture).
- **There’s a clear sweet spot:** **moderate agentic control** (query rewriting + doc grading) sits on the best accuracy/latency frontier.
- **Complex pipelines can be fragile:** wide retrieval + reranking introduces failure modes (esp. with highly quantized models).
- **Quantization fragility is model-dependent:** **4→2-bit** compression can be catastrophic for some models inside complex pipelines (e.g., Llama IQ2), while others degrade modestly.

📄 **Read the write-up:** *(link to your hosted article)*  

---
## What’s in here

### 📊 Analysis & Results
- `index.qmd` — The main article source (Quarto). Renders the figures and narrative.
- `results_table.csv` — Aggregated results summary (one row per configuration).
- `evals/` — Directory containing the raw, per-question JSONL logs for all 96 configurations.

### 🚀 Benchmark Scripts (`scripts/`)
- `scripts/run_baseline.py` — The evaluation harness for Systems A (Baseline), B (Reranked), and C (Agentic).
- `scripts/run_godmode.py` — The evaluation harness for System D ("God Mode" wide-retrieval).
- `scripts/model_factory.py` — Configuration for LLMs and Embedding models (OpenAI, Gemini, Ollama).

### 🛠️ Data Construction & Setup (`setup/`)
- `setup/create_source_data.py` — Scripts to fetch raw movie data via the TMDB API.
- `setup/build_child_parent_db.py` — Generates the ChromaDB vector stores (Child-Parent chunking).
- `setup/gold_set_update.py` — Updates the "Golden Answers" to account for data drift (e.g., changing movie ratings).
- `setup/tmdb_ids_combined.csv` — List of TMDB IDs used to generate the dataset.
- `setup/gold_set_template.jsonl` — The base template used to generate the time-accurate gold set.

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

### Running the Full Benchmark

To reproduce the entire suite of 96 experiments, you can execute the provided batch script:
```bash
run_all_experiments.bat
```

Note: This may take several hours/days depending on your hardware and API limits.

You can also run individual experiments using the CLI arguments:
```bash
python scripts/run_baselines.py --llm gpt4o --db openai_large
```
