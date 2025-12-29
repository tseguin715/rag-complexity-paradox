import argparse
import json
import time
import re
import math
import warnings
import logging
from typing import List, Dict, Any, Optional

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

# Reranker Import
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    print("⚠️  WARNING: sentence-transformers not found. Reranking will be skipped.")

# Import user's model factory
from model_factory import get_embedding_model, get_generative_model

# ==============================================================================
#  CONFIG
# ==============================================================================

CONFIG = {
    "embedding_model": "openai-large",
    "db_path": "db/chroma_db_openai_large",
    "generation_llm": "gpt-4o",
    "judge_llm": "gpt-5.1",
    
    # RERANKER SETTINGS
    "reranker_model": "BAAI/bge-reranker-v2-m3", # SOTA Open Source Reranker
    "retrieval_k": 100,          # Fetch 100 docs (Wide Net)
    "rerank_top_n": 10,          # Keep top 10 (High Precision)
    
    "max_output_tokens": 10240,
}

LLM_ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gemini": "gemini-flash-latest",
    "llama3b": "llama3-8b",
    "gpt-5.1": "gpt-5.1",
    "claude": "claude-sonnet-4-5",
    "gpt4o-mini": "gpt-4o-mini",
    "qwen2.5:32b":"qwen2.5:32b",
    "qwen2.5:32b-q2_K":"qwen2.5:32b-instruct-q2_K",
    "llama3.1:70b":"llama3.1:70b",
    "llama3.1-70b-iq2_xs":"llama3.1-70b-iq2_xs"
}

DB_CONFIGS = {
    "openai_large": {"embedding_model": "openai-large", "db_path": "db/chroma_db_openai_large"},
    "openai_small": {"embedding_model": "openai-small", "db_path": "db/chroma_db_openai_small"},
    "bge_m3": {"embedding_model": "bge-m3", "db_path": "db/chroma_db_bge_m3"},
    "e5_mistral": {"embedding_model": "e5-mistral", "db_path": "db/chroma_db_e5_mistral"},
    "jina_v3": {"embedding_model": "jina-embeddings-v3", "db_path": "db/chroma_db_jina_v3"}
}

# ==============================================================================
#  METRICS & TRACKING UTILS
# ==============================================================================

class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.reset()
    
    def on_llm_end(self, response, **kwargs):
        if not response: return
        prompt_tokens = 0
        completion_tokens = 0
        
        # OpenAI/Anthropic
        if hasattr(response, 'llm_output') and response.llm_output:
            output = response.llm_output
            if isinstance(output, dict):
                usage = output.get('token_usage') or output.get('usage', {})
                if usage:
                    prompt_tokens = usage.get('prompt_tokens') or usage.get('input_tokens', 0)
                    completion_tokens = usage.get('completion_tokens') or usage.get('output_tokens', 0)
        
        # Ollama
        if prompt_tokens == 0 and hasattr(response, 'generations'):
            for gen in response.generations:
                if hasattr(gen[0], 'generation_info'):
                    info = gen[0].generation_info or {}
                    prompt_tokens = info.get('prompt_eval_count', 0)
                    completion_tokens = info.get('eval_count', 0)
                    break
        
        # Gemini - special handling
        if prompt_tokens == 0 and hasattr(response, 'llm_output'):
            # For Gemini, we'll estimate based on response length if not provided
            if response.generations:
                for gen in response.generations:
                    if gen and len(gen) > 0:
                        text = gen[0].text if hasattr(gen[0], 'text') else str(gen[0])
                        # Rough estimation: 1 token ~= 4 chars
                        completion_tokens = max(len(text) // 4, 1)
                        # Estimate prompt tokens (this is very rough)
                        prompt_tokens = 500  # Default estimate
        
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def get_usage(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}

main_token_tracker = TokenUsageTracker()
judge_token_tracker = TokenUsageTracker()

class EvaluationMetrics(BaseModel):
    correctness_score: float = Field(ge=0, le=1)
    faithfulness_score: float = Field(ge=0, le=1)
    justification: str

def apply_max_tokens(llm, max_tokens=10240):
    try:
        # Get class name to identify model type (Ollama, OpenAI, etc.)
        llm_type = str(type(llm)).lower()
        model_name = getattr(llm, 'model_name', '') or getattr(llm, 'model', '')
        
        # 1. OLLAMA: Do NOT bind max_tokens. 
        # Ollama uses 'num_predict', which is already set in your model_factory.py.
        # Binding max_tokens here causes the crash.
        if "ollama" in llm_type:
            return llm
            
        # 2. GEMINI: Uses 'max_output_tokens', not 'max_tokens'
        if "google" in llm_type or "gemini" in str(model_name).lower():
            return llm.bind(max_output_tokens=max_tokens)

        # 3. OPENAI / ANTHROPIC: Use 'max_tokens'
        if 'gpt' in str(model_name).lower() or 'claude' in str(model_name).lower():
            return llm.bind(max_tokens=max_tokens)
            
        # Default: Don't bind to avoid errors on unknown models
        return llm
        
    except Exception as e:
        print(f"⚠️  Could not apply max_tokens: {e}")
        return llm

def calculate_retrieval_metrics(
    retrieved_docs: List[Document],
    gold_doc_ids: List[int],
    question_type: str | None = None,
    k_values=[5, 10],
) -> Dict:
    """Calculate retrieval metrics for a set of retrieved documents."""
    if question_type == "unanswerable":
        return {
            "retrieval_recall": None,
            "retrieval_mrr": None,
            **{f"recall_at_{k}": None for k in k_values},
            **{f"precision_at_{k}": None for k in k_values},
            **{f"ndcg_at_{k}": None for k in k_values},
            **{f"hit_rate_at_{k}": None for k in k_values},
        }
    if not gold_doc_ids or not retrieved_docs:
        metrics = {"retrieval_recall": 0.0, "retrieval_mrr": 0.0}
        for k in k_values:
            metrics.update({f"recall_at_{k}": 0.0, f"precision_at_{k}": 0.0, f"ndcg_at_{k}": 0.0, f"hit_rate_at_{k}": 0.0})
        return metrics
    
    retrieved_ids = [str(d.metadata.get("id")) for d in retrieved_docs if d.metadata.get("id") is not None]
    gold_set = set(str(x) for x in gold_doc_ids)
    
    # Overall recall based on ALL retrieved documents
    overall_recall = 1.0 if gold_set.issubset(set(retrieved_ids)) else 0.0
    
    # MRR calculation
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_set:
            mrr = 1.0 / (i + 1)
            break
            
    metrics = {"retrieval_recall": overall_recall, "retrieval_mrr": mrr}
    
    # Calculate metrics for specific k values
    for k in k_values:
        k_eff = min(k, len(retrieved_ids))
        r_at_k = retrieved_ids[:k_eff]
        r_set_k = set(r_at_k)

        recall_k = len(gold_set.intersection(r_set_k)) / len(gold_set) if gold_set else 0.0
        precision_k = len(gold_set.intersection(r_set_k)) / k_eff if k_eff > 0 else 0.0
        hit_rate_k = 1.0 if len(gold_set.intersection(r_set_k)) > 0 else 0.0

        dcg = sum((1.0 / math.log2(i + 2)) for i, doc_id in enumerate(r_at_k) if doc_id in gold_set)
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(gold_set), k_eff)))
        ndcg_k = dcg / idcg if idcg > 0 else 0.0

        # IMPORTANT: keep the *original* k in the metric key
        metrics[f"recall_at_{k}"] = recall_k
        metrics[f"precision_at_{k}"] = precision_k
        metrics[f"hit_rate_at_{k}"] = hit_rate_k
        metrics[f"ndcg_at_{k}"] = ndcg_k

        
    return metrics

def get_judge_chain(llm):
    parser = JsonOutputParser(pydantic_object=EvaluationMetrics)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate:\n1. Correctness (0-1)\n2. Faithfulness (0-1)\n{format_instructions}\nReturn ONLY valid JSON."),
        ("human", "Q: {question}\nGold: {gold_answer}\nGenerated: {generated_answer}")
    ])
    return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

# ==============================================================================
#  RERANKING LOGIC (CROSS-ENCODER)
# ==============================================================================

# Global cache for the reranker model to avoid reloading
_RERANKER_MODEL = None

def get_reranker_model():
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None and HAS_RERANKER:
        try:
            print(f"    🚀 Loading Reranker: {CONFIG['reranker_model']} (on GPU if avail)...")
            _RERANKER_MODEL = CrossEncoder(CONFIG['reranker_model'], device='cuda', max_length=512)
        except Exception as e:
            print(f"    ❌ Failed to load reranker: {e}")
            return None
    return _RERANKER_MODEL

def cross_encoder_rerank(question: str, docs: List[Document], top_n=10) -> List[Document]:
    """
    Reranks documents using a Cross-Encoder (higher accuracy than vector search).
    Returns reranked docs, or original docs if reranking fails.
    """
    if not docs: 
        return []
    
    # If we have fewer docs than requested, just return them all
    if len(docs) <= top_n:
        return docs
    
    model = get_reranker_model()
    if not model:
        print("    ⚠️  Reranker not available, returning top docs by original order.")
        return docs[:top_n]

    try:
        # Prepare pairs for the Cross Encoder
        pairs = []
        for d in docs:
            content = f"{d.metadata.get('title', '')} {d.page_content}"
            pairs.append([question, content])
        
        # Predict scores
        scores = model.predict(pairs)
        
        # Combine docs with scores
        doc_score_pairs = list(zip(docs, scores))
        
        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        reranked_docs = [d for d, s in doc_score_pairs[:top_n]]
        
        # Debug print top score
        if doc_score_pairs:
            print(f"       🎯 Top Rerank Score: {doc_score_pairs[0][1]:.4f}")
            
        return reranked_docs
    except Exception as e:
        print(f"    ❌ Reranking failed: {e}")
        return docs[:top_n]

# ==============================================================================
#  PIPELINE COMPONENTS
# ==============================================================================

def is_ollama_model(llm) -> bool:
    return type(llm).__name__ == "ChatOllama"

def is_gemini_model(llm) -> bool:
    model_name = str(getattr(llm, 'model_name', '')).lower()
    return 'gemini' in model_name

def contextualize_question(question: str, history: List, llm) -> str:
    if not history:
        return question
    
    # Check if this is an Ollama model for better prompting
    if is_ollama_model(llm):
        system_prompt = """Given a chat history and a user question, rewrite the question to be standalone.
IMPORTANT: 
- Only return the rewritten question
- Do NOT answer the question
- Do NOT ask questions back
- If the question references 'it' or 'this', replace with the actual subject from history
- If no rewriting is needed, return the original question unchanged"""
    else:
        system_prompt = """Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just rewrite it if context is needed."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    # Track tokens
    try:
        result = chain.invoke({"chat_history": history, "input": question}, config={"callbacks": [main_token_tracker]})
    except:
        return question
    
    # Post-process for Ollama to remove any question-asking patterns
    if is_ollama_model(llm):
        if "?" in result and any(phrase in result.lower() for phrase in ["what would you", "what's your", "how can i"]):
            return question
    
    return result

def extract_metadata_filters(question: str) -> Dict[str, Any]:
    filters = {}
    q_lower = question.lower()
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
    if years:
        filters['release_year'] = int(years[-1])
        if "after" in q_lower: filters['release_year'] = {"$gt": int(years[-1])}
        elif "before" in q_lower: filters['release_year'] = {"$lt": int(years[-1])}
        
    lang_map = {"french": "fr", "english": "en", "spanish": "es", "german": "de", "chinese": "zh", "russian": "ru", "korean": "ko", "japanese": "ja"}
    for word in q_lower.split():
        clean = word.strip(".,?!")
        if clean in lang_map:
            filters['original_language'] = lang_map[clean]
            break
            
    if "tv" in q_lower or "series" in q_lower: filters['type'] = "tv"
    elif "movie" in q_lower or "film" in q_lower: filters['type'] = "movie"
    
    return filters

def smart_retrieve(question: str, vector_store, k=50) -> List[Document]:
    """Retrieves a large pool of documents for reranking."""
    filters = extract_metadata_filters(question)
    chroma_filter = {}
    if filters:
        filter_list = [{k: v} for k, v in filters.items()]
        chroma_filter = {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]
    
    print(f"       🔍 Filters: {chroma_filter}")
    
    docs = []
    try:
        if chroma_filter:
            docs = vector_store.similarity_search(question, k=k, filter=chroma_filter)
        else:
            docs = vector_store.similarity_search(question, k=k)
    except Exception as e:
        print(f"       ❌ Retrieval with filters failed: {e}")
        docs = []
            
    # Fallback if filters were too strict
    if len(docs) < 5:
        print("       ⚠️  Low recall, broadening search...")
        try:
            docs_fallback = vector_store.similarity_search(question, k=k)
            seen = {d.metadata.get('id') for d in docs if d.metadata.get('id')}
            for d in docs_fallback:
                if d.metadata.get('id') not in seen:
                    docs.append(d)
                    if len(docs) >= k:
                        break
        except Exception as e:
            print(f"       ❌ Fallback retrieval failed: {e}")
                    
    print(f"       📚 Retrieved {len(docs)} documents")
    return docs

def generate_answer(question: str, docs: List[Document], llm) -> str:
    """Generate answer with improved Ollama support, strict citations, and suppressed internal knowledge."""
    if not docs: 
        return "I cannot find the answer in the provided documents."
    
    context_str = ""
    for d in docs:
        # Retrieve the specific ID from metadata for citation
        doc_id = d.metadata.get('id', 'Unknown')
        meta = f"Title: {d.metadata.get('title', 'N/A')} | Year: {d.metadata.get('release_year', 'N/A')}"
        # Inject [Doc X] into the context so the LLM can reference it
        context_str += f"[Doc {doc_id}] {meta}\n{d.page_content}\n\n"
    
    # Check if this is an Ollama model
    if is_ollama_model(llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided context.

CRITICAL INSTRUCTIONS:
1. Answer the question directly using ONLY information from the context
2. Never ask questions back to the user
3. Cite your sources using [Doc ID] notation when referencing information
4. If the answer is not in the context, say "I cannot find the answer in the provided documents"
5. Be concise and specific
6. Do not use any knowledge from your training data - ONLY use the provided context"""),
            ("human", """Answer this question based ONLY on the context provided.

Question: {question}

Context:
{context}

Direct Answer (with citations):""")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer based strictly on the context. Cite sources using [Doc ID]. If answer is not found, say so. Do not use external knowledge."""),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Generate with retry logic for Ollama
    max_attempts = 3 if is_ollama_model(llm) else 1
    
    for attempt in range(max_attempts):
        try:
            answer = chain.invoke({"question": question, "context": context_str}, 
                                config={"callbacks": [main_token_tracker]})
        except Exception as e:
            print(f"       ❌ Generation failed: {e}")
            return "I cannot generate an answer at this time."
        
        # Post-process for Ollama models
        if is_ollama_model(llm):
            # Check for question-asking patterns
            if any(phrase in answer.lower()[:100] for phrase in 
                   ["what would you", "what's your", "how can i", "could you please", 
                    "can you provide", "would you like", "what specific"]):
                
                if attempt < max_attempts - 1:
                    print(f"       ⚠️  Attempt {attempt+1}: Model asked question, retrying...")
                    continue
                
                # Final cleanup attempt
                sentences = answer.split('.')
                clean_sentences = []
                for sent in sentences:
                    if not any(bad in sent.lower() for bad in 
                             ["what would", "what's your", "how can", "would you", 
                              "could you", "can you", "please provide"]):
                        clean_sentences.append(sent.strip())
                
                if clean_sentences:
                    answer = '.'.join(clean_sentences).strip()
                    if answer and not answer.endswith('.'):
                        answer += '.'
                else:
                    # Fallback if everything is problematic
                    answer = "I cannot determine the answer from the context."
                break
    
    return answer

# ==============================================================================
#  MAIN LOOP - FIXED VERSION
# ==============================================================================

def run_sys_g_v4():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--db", default="openai_large")
    parser.add_argument("--max-tokens", type=int, default=CONFIG["max_output_tokens"])
    args = parser.parse_args()
    
    gen_llm_name = LLM_ALIASES.get(args.llm, args.llm)
    db_conf = DB_CONFIGS[args.db]
    
    print(f"--- STARTING SysG v4 (Cross-Encoder Reranking) - FIXED ---")
    print(f"    Generation: {gen_llm_name}")
    print(f"    Embedding:  {db_conf['embedding_model']}")
    print(f"    Reranker:   {CONFIG['reranker_model']}")
    print(f"    Retrieval K: {CONFIG['retrieval_k']}")
    print(f"    Rerank Top N: {CONFIG['rerank_top_n']}")
    
    embed_model = get_embedding_model(db_conf["embedding_model"])
    gen_llm = apply_max_tokens(get_generative_model(gen_llm_name), args.max_tokens)
    judge_llm = apply_max_tokens(get_generative_model("gpt-5.1"), args.max_tokens)
    
    # Check if this is a Gemini model
    is_gemini = is_gemini_model(gen_llm)
    
    # Initialize Reranker early to check for errors
    if HAS_RERANKER:
        reranker = get_reranker_model()
        if not reranker:
            print("⚠️  WARNING: Reranker failed to load, will proceed without reranking")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector_store = Chroma(persist_directory=db_conf["db_path"], embedding_function=embed_model)
        
    judge_chain = get_judge_chain(judge_llm)
    
    with open("gold_set.jsonl", "r") as f:
        gold_data = [json.loads(line) for line in f]
        
    out_filename = f"evaluation_results_{gen_llm_name}_SysG_v4_Reranked_FIXED_{int(time.time())}.jsonl"
    out_filename = out_filename.replace(':', '_')
    chat_history = []
    
    for q_data in gold_data:
        qid = q_data['question_id']
        raw_question = q_data['question']
        gold_answer = q_data.get('gold_answer', '')
        gold_doc_ids = q_data.get('gold_doc_ids', [])
        q_type = q_data.get('question_type', 'unknown')
        
        print(f"\n[{qid}] {raw_question}")
        
        main_token_tracker.reset()
        judge_token_tracker.reset()
        start_ts = time.time()
        
        # --- PIPELINE ---
        # 1. Contextualize
        question = contextualize_question(raw_question, chat_history, gen_llm)
        
        # 2. Wide Retrieval (k=100 or configured value)
        all_retrieved_docs = smart_retrieve(question, vector_store, k=CONFIG["retrieval_k"])
        
        # IMPORTANT FIX: Calculate retrieval metrics on ALL retrieved docs
        ret_metrics = calculate_retrieval_metrics(all_retrieved_docs, gold_doc_ids, question_type=q_type)
        
        # 3. Cross-Encoder Reranking (Top 10)
        top_docs = cross_encoder_rerank(question, all_retrieved_docs, top_n=CONFIG["rerank_top_n"])
        
        # Calculate post-reranking metrics (optional, for analysis)
        rerank_metrics = calculate_retrieval_metrics(top_docs, gold_doc_ids, question_type=q_type)
        
        # 4. Generate answer using reranked docs
        answer = generate_answer(question, top_docs, gen_llm)
        
        latency = (time.time() - start_ts) * 1000
        
        # --- EVALUATION ---
        try:
            eval_res = judge_chain.invoke({
                "question": raw_question,
                "gold_answer": gold_answer,
                "generated_answer": answer
            }, config={"callbacks": [judge_token_tracker]})
            if hasattr(eval_res, 'dict'): 
                eval_res = eval_res.dict()
        except Exception as e:
            print(f"       ❌ Judge failed: {e}")
            eval_res = {"correctness_score": 0.0, "faithfulness_score": 0.0, "justification": "Error"}

        # Update history
        chat_history.append(HumanMessage(content=raw_question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 4: 
            chat_history = chat_history[-4:]
        
        # Get token usage
        main_usage = main_token_tracker.get_usage()
        judge_usage = judge_token_tracker.get_usage()
        
        # Handle Gemini token reporting
        if is_gemini and main_usage['total_tokens'] == 0:
            # Estimate tokens for Gemini
            main_usage['total_tokens'] = len(str(question) + str(answer)) // 4 + 500
            main_usage['prompt_tokens'] = 500
            main_usage['completion_tokens'] = len(str(answer)) // 4
        
        # Log entry with both retrieval and reranking metrics
        log_entry = {
            "experiment_name": f"SysG_v4_GodMode_{gen_llm_name}_{db_conf['embedding_model']}",
            "question_id": qid,
            "question_type": q_type,
            "question": raw_question,
            "gold_answer": gold_answer,
            "generated_answer": answer,
            "correctness_score": eval_res.get("correctness_score", 0),
            "faithfulness_score": eval_res.get("faithfulness_score", 0),
            "latency_ms": latency,
            
            # Original retrieval metrics (from all 100 docs)
            **ret_metrics,
            
            # Add reranking metrics with prefix
            "rerank_recall_at_10": rerank_metrics.get("recall_at_10", 0),
            "rerank_precision_at_10": rerank_metrics.get("precision_at_10", 0),
            "rerank_ndcg_at_10": rerank_metrics.get("ndcg_at_10", 0),
            
            # Token usage
            "main_total_tokens": main_usage['total_tokens'],
            "main_prompt_tokens": main_usage['prompt_tokens'],
            "main_completion_tokens": main_usage['completion_tokens'],
            "judge_total_tokens": judge_usage['total_tokens'],
            "judge_prompt_tokens": judge_usage['prompt_tokens'],
            "judge_completion_tokens": judge_usage['completion_tokens'],
            
            # Debug info
            "docs_retrieved": len(all_retrieved_docs),
            "docs_after_rerank": len(top_docs),
            "reranker_available": HAS_RERANKER
        }
        
        # Write
        with open(out_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Print summary
        print(f"    ✅ Correct: {eval_res.get('correctness_score', 0):.1f}")

        def fmt(x):
            return "NA" if x is None else f"{x:.2f}"

        print(f"    📊 Retrieval Recall: {fmt(ret_metrics.get('retrieval_recall'))}")
        print(f"    📊 Rerank Recall@10: {fmt(rerank_metrics.get('recall_at_10'))}")


        print(f"    ⏱️  Latency: {latency:.0f}ms")
        print(f"    🔢 Tokens: {main_usage['total_tokens']}")
    
    print(f"\n✅ DONE. Results saved to: {out_filename}")

if __name__ == "__main__":
    run_sys_g_v4()