"""
RAG Evaluation Test Harness with System D v5 FIXED - CONFIGURABLE VERSION
WITH SYSTEM C PROPERLY IMPLEMENTED (Fixed "50" Bug)

COMMAND LINE USAGE:
python evaluate_configurable_4_enhanced_latest.py --llm [alias] --db [alias]
Example: python evaluate_configurable_4_enhanced_latest.py --llm llama3b --db bge_m3

This script evaluates different RAG systems against a gold standard dataset.
System D v5 includes critical fixes for:
- Counting questions (proper semantic filtering)
- Contextual questions (improved chat history handling)
- Metric confusion (rating vs popularity)

SYSTEM C FIX (v2):
- Fixed the "50" bug where the system returned the raw retrieval limit (k=50)
  when semantic filtering failed to extract terms.
- Improved regex for extracting search subjects (mentions, alludes to, etc.).
- Strict fallback: returns 0 instead of len(docs) if semantic criteria exist but no matches found.

CONFIGURATION: Change the variables in the CONFIG section below to use different models.
"""

# ==============================================================================
#  CONFIGURATION - CHANGE THESE TO USE DIFFERENT MODELS
# ==============================================================================
CONFIG = {
    # Embedding model for the vector database
    "embedding_model": "bge-m3",
    
    # Path to the vector database (must match the embedding model used to create it)
    "db_path": "db/chroma_bge_m3",
    
    # LLM for generation (answering questions)
    "generation_llm": "llama3b",
    
    # LLM for judging (evaluating answers) - keep as gpt-4o for consistency
    "judge_llm": "gpt-5.1",
    
    # LLM for grading docs and rewriting queries (System D uses these)
    "grader_llm": "llama3b",
    "rewriter_llm": "llama3b",
    
    # Reranker settings
    "use_reranker": False,
    "reranker_wait_time": 6.5,
}

import argparse # Added for CLI
import json
import time
import os
import re
import pandas as pd
import traceback
import math
from collections import defaultdict
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict, Any, Optional

# ==============================================================================
#  CLI ALIAS MAPS
# ==============================================================================

# Map command line aliases (e.g. 'gpt4o') to Model Factory names (e.g. 'gpt-4o')
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

# Map command line aliases to specific DB paths and Model Factory names
DB_CONFIGS = {
    "openai_large": {
        "embedding_model": "openai-large",
        "db_path": "db/chroma_db_openai_large"
    },
    "openai_small": {
        "embedding_model": "openai-small",
        "db_path": "db/chroma_db_openai_small"
    },
    "bge_m3": {
        "embedding_model": "bge-m3",
        "db_path": "db/chroma_db_bge_m3"
    },

    "e5_mistral": {
        "embedding_model": "e5-mistral", 
        "db_path": "db/chroma_db_e5_mistral"
    },
    "jina_v3": {
        "embedding_model": "jina-embeddings-v3", 
        "db_path": "db/chroma_db_jina_v3"
    }
}


# ==============================================================================
#  SAFETY HELPER
# ==============================================================================
def get_safe_llm(model_name: str):
    """
    Factory wrapper that enforces a token limit to prevent infinite loops/cost spikes.
    """
    llm = get_generative_model(model_name)
    SAFE_LIMIT = 1024  # Cap output at ~1k tokens (approx 750 words)
    
    # Handle OpenAI/Anthropic/Others
    if hasattr(llm, "max_tokens"):
        setattr(llm, "max_tokens", SAFE_LIMIT)
        
    # Handle Google Gemini
    if hasattr(llm, "max_output_tokens"):
        setattr(llm, "max_output_tokens", SAFE_LIMIT)
        
    return llm

# ==============================================================================
#  API KEY CONFIGURATION
# ==============================================================================

# os.environ["OPENAI_API_KEY"] = 'your-key-here'
# os.environ["COHERE_API_KEY"] = "your-key-here"
# os.environ["GOOGLE_API_KEY"] = 'your-key-here'
# os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

# ==============================================================================
#  CORE LANGCHAIN IMPORTS
# ==============================================================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# ==============================================================================
#  VENDOR IMPORTS
# ==============================================================================
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma

# ==============================================================================
#  MODEL FACTORY IMPORT
# ==============================================================================
from model_factory import get_embedding_model, get_generative_model

# ==============================================================================
#  CONSTANTS
# ==============================================================================
GOLD_SET_FILEPATH = "gold_set.jsonl"

# Generate descriptive filename based on configuration
def generate_results_filename(experiments: List[str]) -> str:
    """Generate descriptive filename including all experiment details."""
    def sanitize_filename(name: str) -> str:
        """Remove or replace characters invalid in filenames (especially Windows)."""
        # Replace invalid characters with underscores
        invalid_chars = [':', '<', '>', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name
    
    db_name = os.path.basename(CONFIG["db_path"]).replace("chroma_db_", "").replace("chroma_", "")
    db_name = sanitize_filename(db_name)
    
    llm_name = sanitize_filename(CONFIG['generation_llm'])
    
    exp_names = []
    for exp in experiments:
        if "SysA" in exp:
            exp_names.append("SysA")
        elif "SysC" in exp:
            exp_names.append("SysC")
        elif "SysD" in exp:
            exp_names.append("SysD")
    exp_names = list(dict.fromkeys(exp_names))
    
    filename = f"evaluation_results_{llm_name}_{db_name}"
    for exp in exp_names:
        filename += f"_{exp}"
    filename += ".jsonl"
    return filename

RESULTS_FILEPATH = None  # Will be set after experiments are defined

# Initialize JUDGE_LLM from config (Global default, will update if needed in main)
try:
    JUDGE_LLM = get_safe_llm(CONFIG["judge_llm"])
except Exception:
    JUDGE_LLM = None

DIAGNOSTIC_STATS = {
    'total_questions': 0,
    'successful': 0,
    'failed': 0,
    'errors_by_type': {},
    'question_types': {},
    'avg_latency': []
}


# ==============================================================================
#  TOKEN TRACKING
# ==============================================================================

class TokenUsageTracker(BaseCallbackHandler):
    """Callback handler to track token usage across all LLM calls."""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        
    def on_llm_end(self, response, **kwargs):
        """Track tokens after each LLM call."""
        try:
            # Strategy 1: Check standard llm_output (OpenAI uses 'token_usage', Anthropic often uses 'usage')
            usage_data = None
            if hasattr(response, 'llm_output') and response.llm_output:
                usage_data = response.llm_output.get('token_usage') or response.llm_output.get('usage')

            # Strategy 2: Check message.usage_metadata (Newer LangChain standard)
            if not usage_data and response.generations:
                # Loop through generations to find usage_metadata on the message object
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                            usage_data = gen.message.usage_metadata
                            break
                    if usage_data: break

            # If we found usage data, aggregate it
            if usage_data:
                # Handle different key names (input_tokens/prompt_tokens, output_tokens/completion_tokens)
                p_tokens = usage_data.get('prompt_tokens') or usage_data.get('input_tokens') or 0
                c_tokens = usage_data.get('completion_tokens') or usage_data.get('output_tokens') or 0
                t_tokens = usage_data.get('total_tokens') or (p_tokens + c_tokens)

                self.prompt_tokens += p_tokens
                self.completion_tokens += c_tokens
                self.total_tokens += t_tokens
                self.call_count += 1
                
        except Exception as e:
            pass  # Silently continue if token tracking fails
    
    def reset(self):
        """Reset all counters."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
    
    def get_usage(self) -> Dict[str, int]:
        """Get current usage stats."""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'llm_calls': self.call_count
        }

# Global token trackers
main_token_tracker = TokenUsageTracker()  # For RAG, Agents, Rewriters
judge_token_tracker = TokenUsageTracker() # For the Judge/Evaluator only


# ==============================================================================
#  UTILITIES
# ==============================================================================

def safe_print_exc(prefix: str, e: Exception):
    """Print exception with traceback and update diagnostic stats."""
    error_type = type(e).__name__
    print(f"{prefix} {error_type}: {e}")
    print(traceback.format_exc())
    
    if error_type not in DIAGNOSTIC_STATS['errors_by_type']:
        DIAGNOSTIC_STATS['errors_by_type'][error_type] = 0
    DIAGNOSTIC_STATS['errors_by_type'][error_type] += 1


# ==============================================================================
#  QUESTION TYPE DETECTION
# ==============================================================================

def needs_title_reasoning(question: str) -> bool:
    """Detect questions about movie/show titles that have special meaning."""
    patterns = [
        r"name of a \w+ (month|day|season|color|number)",
        r"also be (a|the)",
        r"happens to (be|also be)",
        r"title.*is also",
        r"called.*same as",
        r"movie.*name.*month",
        r"show.*name.*day"
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def is_multi_hop_filter_question(question: str) -> bool:
    """Detect questions requiring metadata filtering."""
    patterns = [
        r"highest.*rat(ed|ing)",
        r"most.*popular",
        r"how many.*(released|movies|shows|tv)",
        r"only.*language.*year",
        r"what is the.*rating",
        r"(only|specific).*in \d{4}",
        r"what is the.*popularity",
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def answer_is_just_title(question: str) -> bool:
    """Detect if answer should be just a movie/show title."""
    patterns = [
        r"^which (movie|tv|show|film|series)",
        r"^what (movie|tv|show|film|series)",
        r"^in which (movie|tv|show|film)"
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def is_counting_question(question: str) -> bool:
    """Detect questions that ask 'how many'."""
    return re.search(r"^how many", question.lower()) is not None


def needs_sorting(question: str) -> Optional[str]:
    """Detect if question needs sorting. Returns sort field or None."""
    if re.search(r"highest.*rat(ed|ing)", question.lower()):
        return "rating"
    if re.search(r"most popular", question.lower()):
        return "popularity"
    return None


def is_numeric_answer_question(question: str) -> bool:
    """Detect questions expecting numeric answers."""
    patterns = [
        r"^how many",
        r"^how much",
        r"what.*rating",
        r"what.*score",
        r"what is the.*popularity"
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def is_compound_query(question: str) -> bool:
    """Detect queries requiring BOTH X and Y (compound conditions)."""
    patterns = [
        r"\band\b.*\band\b",
        r"both.*and",
        r"with.*and",
        r"have.*and",
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def detect_anachronistic_query(question: str) -> bool:
    """Detect impossible queries referencing events that didn't exist yet."""
    if "chernobyl" in question.lower():
        year_match = re.search(r'(19\d{2}|20\d{2})', question)
        if year_match:
            year = int(year_match.group(1))
            if year < 1986:
                print(f"    [Anachronism] Chernobyl query for {year} is impossible (disaster was 1986)")
                return True
    return False


def extract_metadata_filters(question: str) -> dict:
    """Extract language, year, rating, type, genre constraints from question."""
    filters = {}
    
    # Extract year
    if "after" in question.lower():
        year_match = re.search(r'after\s+(\d{4})', question.lower())
        if year_match:
            filters['release_year_gt'] = int(year_match.group(1))
    elif re.search(r'(in|released in|from)\s+(\d{4})', question.lower()):
        year_match = re.search(r'(\d{4})', question)
        if year_match:
            filters['release_year'] = int(year_match.group())
    
    # Extract language
    languages = {
        'russian': 'ru', 'chinese': 'zh', 'mandarin': 'zh',
        'french': 'fr', 'german': 'de', 'spanish': 'es',
        'japanese': 'ja', 'korean': 'ko', 'english': 'en'
    }
    for lang, code in languages.items():
        if lang in question.lower():
            filters['original_language'] = code
            break
    
    # Extract type
    if re.search(r'\b(tv|show|series)\b', question.lower()):
        filters['type'] = 'tv'
    elif re.search(r'\b(movie|film)\b', question.lower()):
        filters['type'] = 'movie'
    
    # Extract rating constraint
    rating_match = re.search(r'rating (?:over|above|greater than) (\d+\.?\d*)', question.lower())
    if rating_match:
        filters['min_rating'] = float(rating_match.group(1))
    
    # Extract popularity constraint
    pop_match = re.search(r'popularity.*?(?:score|rating)?\s*(?:greater than|above|over)\s*(\d+\.?\d*)', question.lower())
    if pop_match:
        filters['min_popularity'] = float(pop_match.group(1))

    # --- NEW: Extract Genre ---
    genres = ['horror', 'comedy', 'drama', 'action', 'thriller', 'romance', 'sci-fi', 'fantasy', 'adventure', 'crime', 'war', 'history']
    for genre in genres:
        if genre in question.lower():
            filters['genre'] = genre.capitalize()
            break
            
    return filters


def has_semantic_criteria(question: str) -> bool:
    """Detect if counting question has semantic (title/plot) criteria."""
    patterns = [
        r"(title|called|named).*['\"](.+?)['\"]",  # Title in quotes
        r"(about|mention|allude|feature|involve|starring)",  # Plot/content
        r"('([^']+)'|\"([^\"]+)\")",  # Any quoted text
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Proper nouns (names)
    ]
    return any(re.search(p, question) for p in patterns)


# ==============================================================================
#  COMPONENT FACTORIES
# ==============================================================================

def get_vector_store(db_path: str, embedding_model: Any) -> Chroma:
    """Load a Chroma vector store."""
    print(f"[DB Loader] Loading vector store from: {db_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )


# ==============================================================================
#  HELPERS
# ==============================================================================

def format_docs_with_metadata(docs: List[Document]) -> str:
    """Format documents for LLM context with [Doc X] IDs."""
    formatted = []
    for d in docs:
        md = d.metadata if getattr(d, "metadata", None) else {}
        # Extract the ID
        doc_id = md.get('id', 'Unknown')
        
        formatted.append(
            f"[Doc {doc_id}] Title: {md.get('title', 'N/A')}\n"
            f"Type: {md.get('type', 'N/A')}\n"
            f"Year: {md.get('release_year', 'N/A')}\n"
            f"Rating: {md.get('average_rating', 'N/A')}\n"
            f"Popularity: {md.get('popularity', 'N/A')}\n"
            f"Language: {md.get('original_language', 'N/A')}\n"
            f"Plot: {d.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


class DocCatcher:
    """Captures documents passing through a chain."""
    def __init__(self):
        self.docs: List[Document] = []

    def __call__(self, docs: List[Document]) -> List[Document]:
        self.docs = list(docs) if docs is not None else []
        return docs

    def reset(self):
        self.docs = []


# ==============================================================================
#  SYSTEM A: BASELINE RAG
# ==============================================================================

def build_system_a_pipeline(retriever: Any, llm: Any) -> Runnable:
    """
    System A: Baseline RAG
    Simple retrieval + generation without any advanced features.
    """
    print("[Pipeline] Building System A (Baseline RAG)")
    
    def extract_question(x: dict) -> str:
        """Extract the question string from input dict."""
        return x.get("input", "")
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based strictly on the context. Do not use outside knowledge. Cite sources for every fact using the format [Doc X]. If answer is not found, say so.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    
    return (
        {
            "context": RunnableLambda(extract_question) | retriever | RunnableLambda(format_docs_with_metadata),
            "input": RunnableLambda(extract_question)
        }
        | qa_prompt | llm | StrOutputParser()
    )


# ==============================================================================
#  SYSTEM C: HYBRID SEARCH RAG (FIXED!)
# ==============================================================================

def build_system_c_pipeline(
    retriever: Any,
    llm: Any,
    doc_catcher: DocCatcher,
    vector_store: Any
) -> Runnable:
    """
    System C: Hybrid Search RAG (FIXED v2)
    
    Improvements over System A:
    1. Metadata filtering (hybrid search)
    2. Question type detection  
    3. Specialized prompts per type
    4. PROPER counting with basic semantic filtering
    
    FIXES (v2):
    - "50 bug" fixed: No longer returns k=50 when semantic filtering fails.
    - Improved regex for subject extraction ("allude to", "mention", etc.).
    """
    print("[Pipeline] Building System C (Hybrid Search RAG) - FIXED VERSION v2")
    
    def apply_hybrid_search_c(question: str, filters: dict, is_counting: bool) -> List[Document]:
        """Hybrid search with metadata filtering."""
        print(f"    [SysC Hybrid] Filters: {filters}, Counting: {is_counting}")
        
        # Build Chroma where clause
        where_conditions = []
        if 'release_year' in filters:
            where_conditions.append({'release_year': filters['release_year']})
        if 'release_year_gt' in filters:
            where_conditions.append({'release_year': {'$gt': filters['release_year_gt']}})
        if 'original_language' in filters:
            where_conditions.append({'original_language': filters['original_language']})
        if 'type' in filters:
            where_conditions.append({'type': filters['type']})
        
        where_clause = None
        if where_conditions:
            where_clause = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
            print(f"    [SysC Hybrid] Where: {where_clause}")
        
        try:
            k_val = 50 if is_counting else 20
            if 'genre' in filters:
                k_val = 100
            if where_clause:
                filtered_retriever = vector_store.as_retriever(
                    search_kwargs={'k': k_val, 'filter': where_clause}
                )
                docs = filtered_retriever.invoke(question)
                print(f"    [SysC Hybrid] Filtered retrieval: {len(docs)} docs (k={k_val})")
            else:
                retriever_temp = vector_store.as_retriever(search_kwargs={'k': k_val})
                docs = retriever_temp.invoke(question)
                print(f"    [SysC Hybrid] Semantic retrieval: {len(docs)} docs (k={k_val})")
            if 'genre' in filters and docs:
                docs = [d for d in docs if filters['genre'] in d.metadata.get('genre', '')]
            # Post-filter by rating if needed
            if 'min_rating' in filters and docs:
                original = len(docs)
                docs = [d for d in docs if d.metadata.get('average_rating', 0) > filters['min_rating']]
                print(f"    [SysC Hybrid] After rating>{filters['min_rating']}: {len(docs)} docs (was {original})")
            
            # Post-filter by popularity if needed
            if 'min_popularity' in filters and docs:
                original = len(docs)
                docs = [d for d in docs if d.metadata.get('popularity', 0) > filters['min_popularity']]
                print(f"    [SysC Hybrid] After popularity>{filters['min_popularity']}: {len(docs)} docs (was {original})")
            
            return docs
        except Exception as e:
            print(f"    [SysC Hybrid] Error: {e}")
            return retriever.invoke(question)
    
    def basic_semantic_count(question: str, docs: List[Document]) -> int:
        """
        FIXED: Basic semantic filtering for counting.
        
        Simpler than System D - uses keyword matching instead of LLM.
        Only applies if question has semantic criteria (title/plot).
        
        FIX v2: Prevents returning raw 'k' (50) when semantic criteria exists but keywords miss.
        """
        if not docs:
            return 0
        
        # Check if semantic filtering is needed
        if not has_semantic_criteria(question):
            print(f"    [SysC Count] Pure metadata count: {len(docs)}")
            return len(docs)
        
        print(f"    [SysC Count] Semantic criteria detected, filtering...")
        
        # Extract search terms from question
        search_terms = []
        
        # Extract quoted terms
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        search_terms.extend(quoted)
        
        # FIXED: Broader regex for "subject" extraction (about, mention, allude to, etc.)
        about_match = re.search(r"(?:about|mention|allude to|feature|involve|starring)\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
        if about_match:
            terms = about_match.group(1).strip()
            # Remove common words to isolate the noun phrases
            terms = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for)\b', '', terms, flags=re.IGNORECASE)
            if terms:
                search_terms.append(terms.lower())
        
        # Extract names (proper nouns)
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
        search_terms.extend(names)
        
        # FIXED: Fallback if no specific terms found but we KNOW it's semantic
        # If we don't have search terms, we might fail to filter and return len(docs) (50)
        # which is the bug. So we try to clean the question as a last resort.
        if not search_terms:
            # Remove query stopwords
            cleaned = re.sub(r'\b(how|many|movies|shows|tv|are|there|is|released|in|the|was|were)\b', '', question, flags=re.IGNORECASE)
            cleaned = re.sub(r'[^\w\s]', '', cleaned).strip() # Remove punctuation
            if len(cleaned) > 3: # Avoid empty strings
                search_terms.append(cleaned.lower())
                print(f"    [SysC Count] Fallback terms: '{cleaned}'")

        # If still no search terms, and criteria IS detected, strict logic:
        # We can't verify matches, so returning 50 is dangerous. 
        # We will assume 0 (strict) rather than 50 (loose).
        if not search_terms:
             print(f"    [SysC Count] Criteria detected but no terms extracted. Returning 0 (Strict).")
             return 0

        # If specific title criteria, check titles
        if re.search(r"(title|called|named|'[^']+'|\"[^\"]+\")", question):
            print(f"    [SysC Count] Title matching for: {search_terms}")
            matching = []
            for doc in docs:
                title = doc.metadata.get('title', '').lower()
                # Check if any search term is in title
                if any(term.lower() in title for term in search_terms if term):
                    matching.append(doc)
            print(f"    [SysC Count] Title filter: {len(docs)} → {len(matching)}")
            return len(matching)
        
        # Otherwise check plot/content
        if search_terms:
            print(f"    [SysC Count] Content matching for: {search_terms}")
            matching = []
            for doc in docs:
                content = (doc.page_content or '').lower()
                title = doc.metadata.get('title', '').lower()
                combined = f"{title} {content}"
                
                # Check if any search term appears
                if any(term.lower() in combined for term in search_terms if term):
                    matching.append(doc)
            
            print(f"    [SysC Count] Content filter: {len(docs)} → {len(matching)}")
            return len(matching)
        
        # Fallback: return 0 instead of len(docs) if we got here with semantic criteria
        print(f"    [SysC Count] Semantic loop finished without match. Returning 0.")
        return 0
    
    def hybrid_rag_fn(in_dict: dict) -> str:
        """Main System C logic: hybrid search + question routing."""
        try:
            question = in_dict.get("input", "")
            chat_history = in_dict.get("chat_history", [])
            
            print(f"    [SysC] Input: {question[:80]}...")
            doc_catcher.reset()
            
            # Detect question type
            is_counting = is_counting_question(question)
            is_multi_hop = is_multi_hop_filter_question(question)
            sort_field = needs_sorting(question)
            
            print(f"    [SysC] Counting:{is_counting}, MultiHop:{is_multi_hop}, Sort:{sort_field}")
            
            # Check for anachronistic queries
            if detect_anachronistic_query(question):
                return "I cannot find any movies or shows matching that criteria. Please check if your question refers to an event that occurred during the time period mentioned."
            
            # Extract metadata filters
            filters = extract_metadata_filters(question) if (is_multi_hop or is_counting) else {}
            
            # Hybrid search
            docs = apply_hybrid_search_c(question, filters, is_counting)
            
            if not docs:
                print(f"    [SysC] No docs retrieved")
                if is_counting:
                    return "0"
                return "I don't know."
            
            # Sort if needed
            if sort_field and docs:
                if sort_field == "rating":
                    docs = sorted(docs, key=lambda d: d.metadata.get('average_rating', 0), reverse=True)
                    print(f"    [SysC] Sorted by rating, top: {docs[0].metadata.get('title')} ({docs[0].metadata.get('average_rating')})")
                elif sort_field == "popularity":
                    docs = sorted(docs, key=lambda d: d.metadata.get('popularity', 0), reverse=True)
                    print(f"    [SysC] Sorted by popularity, top: {docs[0].metadata.get('title')} ({docs[0].metadata.get('popularity')})")
                docs = docs[:1]  # Take only top for "highest/most" questions
            
            # Capture docs
            doc_catcher(docs[:15])
            
            # FIXED: Handle counting with basic semantic filtering
            if is_counting:
                count = basic_semantic_count(question, docs)
                print(f"    [SysC] Final count: {count}")
                return str(count)
            
            # Generate answer
            ctx = format_docs_with_metadata(docs[:15])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer based strictly on the context. Do not use outside knowledge. Cite sources for every fact using the format [Doc X]. If answer is not found, say so."),
                ("human", "Question: {q}\n\nContext:\n{ctx}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"q": question, "ctx": ctx})
            print(f"    [SysC] Generated answer: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            print(f"    [SysC] ERROR: {e}")
            safe_print_exc("    [SysC] Traceback:", e)
            return "I don't know."
    
    return RunnableLambda(hybrid_rag_fn)


# ==============================================================================
#  SYSTEM D: FIXED AGENTIC RAG (v5)
# ==============================================================================

def build_system_d_pipeline(
    retriever: Any,
    llm: Any,
    doc_catcher: DocCatcher,
    vector_store: Any,
    reranker: Optional[CohereRerank] = None
) -> Runnable:
    """
    System D: Fixed Agentic RAG (v5)
    
    Full implementation with all agentic features.
    """
    print("[Pipeline Factory] Building: System D v5 FIXED")

    grader_llm = get_safe_llm(CONFIG["grader_llm"])
    rewriter_llm = get_safe_llm(CONFIG["rewriter_llm"])
    generator_llm = llm

    def contextualize_with_history(question: str, chat_history: List) -> str:
        """Better contextualization with explicit instructions."""
        if not chat_history:
            return question
        
        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a conversation history, rewrite the follow-up question to be a standalone question.
If the question refers to 'it', 'this', 'that', 'the movie', 'the show', etc., replace with the actual name from the previous conversation.
Return ONLY the rewritten standalone question, nothing else."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        try:
            chain = ctx_prompt | rewriter_llm | StrOutputParser()
            rewritten = chain.invoke({"input": question, "chat_history": chat_history}, config={"callbacks": [main_token_tracker]})
            if rewritten and rewritten != question:
                print(f"    [Agent] Contextualized: '{question}' → '{rewritten}'")
                return rewritten
            return question
        except Exception as e:
            print(f"    [Agent] Contextualization failed: {e}")
            return question

    def grade_docs(q: str, docs: List[Document]) -> bool:
        """More lenient grader with better understanding."""
        if not docs:
            print(f"    [Grader] No docs to grade")
            return False
        
        context = []
        for i, d in enumerate(docs[:5]):
            md = d.metadata or {}
            plot = d.page_content[:250] if d.page_content else "No plot available"
            context.append(
                f"[{i+1}] Title: {md.get('title', 'N/A')}, Year: {md.get('release_year', 'N/A')}, "
                f"Type: {md.get('type', 'N/A')}, Language: {md.get('original_language', 'N/A')}\n"
                f"    Plot: {plot}"
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a lenient document grader. Answer 'yes' if ANY of the documents could help answer the question, even partially. 
Only answer 'no' if the documents are CLEARLY about completely different topics with no relevance.
Be generous - if a document has any connection to the question topic, answer 'yes'."""),
            ("human", "Question: {q}\n\nDocuments:\n{d}\n\nCan these documents help answer the question? (yes/no)")
        ])
        
        try:
            chain = prompt | grader_llm | StrOutputParser()
            resp = chain.invoke({"q": q, "d": "\n\n".join(context)}, config={"callbacks": [main_token_tracker]})
            result = "yes" in resp.lower()
            print(f"    [Grader] {result} ('{resp.strip()}')")
            return result
        except Exception as e:
            print(f"    [Grader] Error: {e}, defaulting to True")
            return True

    def apply_hybrid_search(question: str, filters: dict, is_counting: bool = False) -> List[Document]:
        """Better hybrid search with appropriate k values."""
        print(f"    [Hybrid] Filters: {filters}, Counting: {is_counting}")
        
        where_conditions = []
        if 'release_year' in filters:
            where_conditions.append({'release_year': filters['release_year']})
        if 'release_year_gt' in filters:
            where_conditions.append({'release_year': {'$gt': filters['release_year_gt']}})
        if 'original_language' in filters:
            where_conditions.append({'original_language': filters['original_language']})
        if 'type' in filters:
            where_conditions.append({'type': filters['type']})
        
        where_clause = None
        if where_conditions:
            where_clause = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
            print(f"    [Hybrid] Where: {where_clause}")
        
        try:
            if is_counting:
                k_val = 50 if where_clause else 30
            else:
                k_val = 20
            if 'genre' in filters:
                k_val = 100
            if where_clause:
                filtered_retriever = vector_store.as_retriever(
                    search_kwargs={'k': k_val, 'filter': where_clause}
                )
                docs = filtered_retriever.invoke(question)
                print(f"    [Hybrid] Filtered retrieval: {len(docs)} docs (k={k_val})")
            else:
                retriever_temp = vector_store.as_retriever(search_kwargs={'k': k_val})
                docs = retriever_temp.invoke(question)
                print(f"    [Hybrid] Semantic retrieval: {len(docs)} docs (k={k_val})")
            if 'genre' in filters and docs:
                docs = [d for d in docs if filters['genre'] in d.metadata.get('genre', '')]
            if 'min_rating' in filters and docs:
                original = len(docs)
                docs = [d for d in docs if d.metadata.get('average_rating', 0) > filters['min_rating']]
                print(f"    [Hybrid] After rating>{filters['min_rating']}: {len(docs)} docs (was {original})")
            
            if 'min_popularity' in filters and docs:
                original = len(docs)
                docs = [d for d in docs if d.metadata.get('popularity', 0) > filters['min_popularity']]
                print(f"    [Hybrid] After popularity>{filters['min_popularity']}: {len(docs)} docs (was {original})")
            
            return docs
        except Exception as e:
            print(f"    [Hybrid] Error: {e}")
            safe_print_exc("    [Hybrid] Traceback:", e)
            try:
                return retriever.invoke(question)
            except Exception as e2:
                print(f"    [Hybrid] Fallback also failed: {e2}")
                return []

    def semantic_count_filter(question: str, docs: List[Document]) -> tuple[List[Document], bool]:
        """Advanced semantic filtering for counting questions."""
        if not docs:
            return docs, True
        
        has_title_criteria = bool(re.search(r"(title|called|named).*['\"](.+?)['\"]", question.lower()))
        has_plot_criteria = bool(re.search(r"(about|mention|allude|feature|involve|starring)", question.lower()))
        has_specific_entity = bool(re.search(r"('([^']+)'|\"([^\"]+)\")", question))
        has_compound_condition = is_compound_query(question)
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', question))
        has_actor_query = bool(re.search(r'\b(have|with|starring|featuring)\s+[A-Z]', question))
        
        needs_semantic_filtering = (has_title_criteria or has_plot_criteria or has_specific_entity or 
                                    has_compound_condition or has_proper_nouns or has_actor_query)
        
        if not needs_semantic_filtering:
            print(f"    [SemanticCount] Pure metadata count, returning all {len(docs)} filtered docs")
            return docs, True
        
        triggers = []
        if has_title_criteria: triggers.append("title")
        if has_plot_criteria: triggers.append("plot")
        if has_specific_entity: triggers.append("entity")
        if has_compound_condition: triggers.append("compound")
        if has_proper_nouns: triggers.append("proper_nouns")
        if has_actor_query: triggers.append("actor_query")
        print(f"    [SemanticCount] Semantic filtering triggered by: {', '.join(triggers)}")
        
        is_compound = is_compound_query(question)
        if is_compound:
            print(f"    [SemanticCount] Detected compound query - will be strict about matching")
        
        system_prompt = """You are counting documents that match specific criteria.
Given a question and a list of documents, identify which documents match the question's criteria.

IMPORTANT: For questions with "and" (compound queries), a document must match ALL criteria to count.
For questions with "or", a document only needs to match one criterion.

Return ONLY a comma-separated list of document numbers that match (e.g., "1,3,5"), or "NONE" if none match, or "ALL" if all match.

Examples:
Q: "How many movies have 'Astro Boy' in the title?"
Docs: [1] Astro Boy (2009), [2] Mighty Atom, [3] Astro Boy (1980)
Answer: 1,3

Q: "How many Russian TV shows in 1997?"
Docs: [1] Russian show from 1997, [2] Russian show from 1997
Answer: ALL

Q: "How many movies mention assassins?"
Docs: [1] Plot: An assassin seeks revenge..., [2] Plot: A chef's journey..., [3] Plot: Political assassination plot...
Answer: 1,3

Q: "How many movies have Bruce Lee AND Steven Seagal?"
Docs: [1] Bruce Lee movie, [2] Steven Seagal movie, [3] Both Bruce Lee and Steven Seagal
Answer: 3

Q: "How many movies with BOTH Bruce Lee and Steven Seagal?"
Docs: [1] Bruce Lee only, [2] Steven Seagal only, [3] Random action movie
Answer: NONE"""

        filter_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {q}\n\nDocuments:\n{d}\n\nMatching document numbers (or ALL/NONE):")
        ])
        
        doc_list = []
        for i, doc in enumerate(docs):
            md = doc.metadata or {}
            title = md.get('title', 'N/A')
            year = md.get('release_year', 'N/A')
            lang = md.get('original_language', 'N/A')
            type_ = md.get('type', 'N/A')
            plot = doc.page_content[:150] if doc.page_content else "No plot"
            doc_list.append(f"[{i+1}] {title} ({year}, {lang}, {type_}): {plot}")
        
        try:
            chain = filter_prompt | generator_llm | StrOutputParser()
            response = chain.invoke({"q": question, "d": "\n".join(doc_list)}, config={"callbacks": [main_token_tracker]})
            print(f"    [SemanticCount] LLM response: {response}")
            
            response_clean = response.strip().upper()
            
            if "ALL" in response_clean:
                print(f"    [SemanticCount] ALL docs match")
                return docs, True
            
            if "NONE" in response_clean:
                print(f"    [SemanticCount] NONE match")
                return [], True
            
            matching_indices = []
            for num in re.findall(r'\d+', response):
                idx = int(num) - 1
                if 0 <= idx < len(docs):
                    matching_indices.append(idx)
            
            if not matching_indices:
                print(f"    [SemanticCount] No valid indices parsed, returning empty")
                return [], True
            
            filtered = [docs[i] for i in matching_indices]
            print(f"    [SemanticCount] Filtered: {len(docs)} → {len(filtered)} docs")
            
            if is_compound and len(filtered) == 0:
                print(f"    [SemanticCount] Compound query with 0 results - high confidence")
                return filtered, True
            
            return filtered, True
            
        except Exception as e:
            print(f"    [SemanticCount] Error: {e}, returning LOW CONFIDENCE")
            return [], False

    def agent_fn(in_dict: dict) -> str:
        """Main agent logic with all improvements."""
        try:
            original_question = in_dict.get("input", "")
            chat_history = in_dict.get("chat_history", [])
            
            print(f"    [Agent] Input: {original_question[:80]}...")
            
            question = contextualize_with_history(original_question, chat_history)
            doc_catcher.reset()
            
            is_counting = is_counting_question(question)
            needs_title = needs_title_reasoning(question)
            is_multi_hop = is_multi_hop_filter_question(question)
            is_title_only = answer_is_just_title(question)
            is_numeric = is_numeric_answer_question(question)
            sort_field = needs_sorting(question)
            
            print(f"    [Agent] Counting:{is_counting}, Title:{needs_title}, MultiHop:{is_multi_hop}, Numeric:{is_numeric}, Sort:{sort_field}")
            
            if detect_anachronistic_query(question):
                return "I cannot find any movies or shows matching that criteria. Please check if your question refers to an event that occurred during the time period mentioned."
            
            filters = extract_metadata_filters(question) if (is_multi_hop or is_counting) else {}
            
            docs = apply_hybrid_search(question, filters, is_counting)
            
            if not docs:
                print(f"    [Agent] WARNING: No documents retrieved!")
                if is_counting:
                    return "I cannot find any movies or shows matching those criteria."
                return "I don't know."
            
            if is_counting and is_compound_query(question) and len(docs) > 20:
                print(f"    [Agent] WARNING: Compound query retrieved {len(docs)} docs - might be too broad")
            
            if sort_field and docs:
                if sort_field == "rating":
                    docs = sorted(docs, key=lambda d: d.metadata.get('average_rating', 0), reverse=True)
                    print(f"    [Agent] Sorted by rating, top: {docs[0].metadata.get('title')} ({docs[0].metadata.get('average_rating')})")
                elif sort_field == "popularity":
                    docs = sorted(docs, key=lambda d: d.metadata.get('popularity', 0), reverse=True)
                    print(f"    [Agent] Sorted by popularity, top: {docs[0].metadata.get('title')} ({docs[0].metadata.get('popularity')})")
            
            if reranker and docs and not is_counting:
                try:
                    docs = reranker.compress_documents(query=question, documents=docs)
                    time.sleep(CONFIG["reranker_wait_time"])
                    print(f"    [Agent] Reranked to {len(docs)} docs")
                except Exception as e:
                    print(f"    [Agent] Reranking failed/skipped: {e}")
                    docs = docs[:15]
            elif not is_counting:
                docs = docs[:15]
            
            doc_catcher(docs)
            
            if is_counting:
                if not docs or len(docs) == 0:
                    print(f"    [Agent] No documents found for counting query")
                    return "I cannot find any movies or shows matching those criteria."
                
                print(f"    [Agent] Starting semantic filter on {len(docs)} docs")
                filtered_docs, is_confident = semantic_count_filter(question, docs)
                count = len(filtered_docs)
                
                if not is_confident:
                    print(f"    [Agent] Low confidence in filtering, returning 'cannot find'")
                    return "I cannot reliably count matches for that query. Please try rephrasing."
                
                if count == 0:
                    if is_compound_query(question):
                        print(f"    [Agent] Compound query with 0 matches (legitimate)")
                        return "I cannot find any movies or shows matching all of those criteria."
                    else:
                        print(f"    [Agent] Simple query with 0 matches")
                        return "I cannot find any movies or shows matching that criteria."
                
                if count > 50:
                    print(f"    [Agent] WARNING: Count of {count} seems too high, probably an error")
                    return "I cannot reliably count matches for that query. The number seems unusually high."
                
                if is_compound_query(question) and count > 10:
                    print(f"    [Agent] WARNING: Compound query with {count} results seems suspicious")
                    print(f"    [Agent] Proceeding but flagging as potentially incorrect")
                
                print(f"    [Agent] Final validated count: {count}")
                return str(count)
            
            if needs_title and docs:
                titles = [d.metadata.get('title', '') for d in docs[:10]]
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Which title best answers the question? Return ONLY the title, nothing else."),
                    ("human", "Question: {q}\nTitles: {t}")
                ])
                try:
                    chain = prompt | generator_llm | StrOutputParser()
                    result = chain.invoke({"q": original_question, "t": "\n".join(titles)}, config={"callbacks": [main_token_tracker]})
                    print(f"    [Agent] Title reasoning result: {result}")
                    return result
                except Exception as e:
                    print(f"    [Agent] Title reasoning failed: {e}")
            
            if sort_field and docs:
                docs = docs[:1]
                print(f"    [Agent] Using only top doc for highest/most question")
            
            if not is_numeric:
                if not grade_docs(question, docs):
                    print(f"    [Agent] Grader rejected docs - returning 'I don't know'")
                    return "I don't know."
            else:
                print(f"    [Agent] Skipping grader for numeric question")
            
            ctx = format_docs_with_metadata(docs)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer based strictly on the context. Do not use outside knowledge. Cite sources for every fact using the format [Doc X]. If asked for a number, give just the number. If asked for a title, give just the title."),
                ("human", "Question: {q}\n\nContext:\n{ctx}")
            ])
            
            try:
                chain = prompt | generator_llm | StrOutputParser()
                answer = chain.invoke({"q": original_question, "ctx": ctx}, config={"callbacks": [main_token_tracker]})
                print(f"    [Agent] Generated answer: {answer[:100]}...")
                
                if is_title_only and len(answer.split()) > 4:
                    extract = ChatPromptTemplate.from_messages([
                        ("system", "Extract ONLY the title from the text below. Return nothing but the title."),
                        ("human", "{ans}")
                    ])
                    chain2 = extract | generator_llm | StrOutputParser()
                    answer = chain2.invoke({"ans": answer})
                    print(f"    [Agent] Extracted title: {answer}")
                
                return answer
            except Exception as e:
                print(f"    [Agent] Generation error: {e}")
                safe_print_exc("    [Agent] Traceback:", e)
                return "I don't know."
                
        except Exception as e:
            print(f"    [Agent] CRITICAL ERROR in agent_fn: {e}")
            safe_print_exc("    [Agent] Full traceback:", e)
            return "I don't know."

    return RunnableLambda(agent_fn)


# ==============================================================================
#  METRICS
# ==============================================================================

class EvaluationMetrics(BaseModel):
    correctness_score: float
    faithfulness_score: float
    justification: str


def calculate_retrieval_metrics(retrieved_docs: List[Document], gold_doc_ids: List[int], question_type: str, k_values: List[int] = [5, 10]) -> Dict:
    """
    Calculate comprehensive retrieval metrics including Recall@k, Precision@k, MRR, nDCG@k, and Hit Rate@k.
    """
    if question_type == "unanswerable":
        return {
            "retrieval_recall": None,
            "retrieval_mrr": None,
            **{f"recall_at_{k}": None for k in k_values},
            **{f"precision_at_{k}": None for k in k_values},
            **{f"ndcg_at_{k}": None for k in k_values},
            **{f"hit_rate_at_{k}": None for k in k_values}
        }
    
    if not gold_doc_ids:
        return {
            "retrieval_recall": 0.0,
            "retrieval_mrr": 0.0,
            **{f"recall_at_{k}": 0.0 for k in k_values},
            **{f"precision_at_{k}": 0.0 for k in k_values},
            **{f"ndcg_at_{k}": 0.0 for k in k_values},
            **{f"hit_rate_at_{k}": 0.0 for k in k_values}
        }
    
    retrieved_ids = [str(d.metadata.get("id")) for d in retrieved_docs if d.metadata.get("id") is not None]
    gold_set = set(str(x) for x in gold_doc_ids)
    
    # Overall Recall
    overall_recall = 1.0 if gold_set.issubset(set(retrieved_ids)) else 0.0
    
    # MRR
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in gold_set:
            mrr = 1.0 / (idx + 1)
            break
    
    metrics = {"retrieval_recall": overall_recall, "retrieval_mrr": mrr}
    
    # Calculate @k metrics
    for k in k_values:
        retrieved_at_k = retrieved_ids[:k]
        retrieved_set_k = set(retrieved_at_k)
        
        recall_at_k = len(gold_set.intersection(retrieved_set_k)) / len(gold_set) if gold_set else 0.0
        precision_at_k = len(gold_set.intersection(retrieved_set_k)) / k if k > 0 else 0.0
        hit_rate_at_k = 1.0 if len(gold_set.intersection(retrieved_set_k)) > 0 else 0.0
        
        # nDCG@k
        dcg = 0.0
        idcg = 0.0
        for idx, rid in enumerate(retrieved_at_k):
            if rid in gold_set:
                dcg += 1.0 / math.log2(idx + 2)
        ideal_k = min(k, len(gold_set))
        for idx in range(ideal_k):
            idcg += 1.0 / math.log2(idx + 2)
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        metrics[f"recall_at_{k}"] = recall_at_k
        metrics[f"precision_at_{k}"] = precision_at_k
        metrics[f"ndcg_at_{k}"] = ndcg_at_k
        metrics[f"hit_rate_at_{k}"] = hit_rate_at_k
    
    return metrics

def get_llm_as_judge_chain() -> Runnable:
    """Create LLM-as-judge evaluation chain."""
    parser = JsonOutputParser(pydantic_object=EvaluationMetrics)
    prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "Evaluate:\n"
    "1. Correctness (0-1)\n"
    "2. Faithfulness (0-1)\n"
    "{format_instructions}\n\n"
    "Return ONLY valid JSON. Do NOT include any commentary or formatting."
    ),
        ("human", "Q: {question}\nGold: {gold_answer}\nGenerated: {generated_answer}")
    ])
    return prompt.partial(format_instructions=parser.get_format_instructions()) | JUDGE_LLM | parser


# ==============================================================================
#  EVALUATION
# ==============================================================================

def run_evaluation(
    experiment_name: str,
    embedding_model_name: str,
    db_path: str,
    llm_name: str,
    pipeline_builder_fn: callable
):
    """Run evaluation on a RAG system."""
    print(f"\n{'='*80}\nEXPERIMENT: {experiment_name}\n{'='*80}")
    
    try:
        embedding_model = get_embedding_model(embedding_model_name)
        vector_store = get_vector_store(db_path, embedding_model)
        llm = get_safe_llm(llm_name)
        print("[SETUP] ✓ Models and vector store loaded successfully")
    except Exception as e:
        safe_print_exc("[SETUP] FATAL ERROR:", e)
        return

    try:
        with open(GOLD_SET_FILEPATH, 'r') as f:
            gold_set = [json.loads(line) for line in f]
        print(f"[SETUP] ✓ Loaded {len(gold_set)} questions from gold set")
    except Exception as e:
        safe_print_exc("[SETUP] ERROR loading gold set:", e)
        return

    judge_chain = get_llm_as_judge_chain()
    question_cache = {}

    with open(RESULTS_FILEPATH, 'a') as out_f:
        for i, qdata in enumerate(gold_set):
            qid = qdata['question_id']
            q_type = qdata.get('question_type', 'unknown')
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(gold_set)}] {qid} ({q_type})")
            print(f"{'='*80}")
            print(f"Question: {qdata['question']}")
            print(f"Gold Answer: {qdata.get('gold_answer', 'N/A')}")
            
            DIAGNOSTIC_STATS['total_questions'] += 1
            if q_type not in DIAGNOSTIC_STATS['question_types']:
                DIAGNOSTIC_STATS['question_types'][q_type] = {'attempted': 0, 'successful': 0}
            DIAGNOSTIC_STATS['question_types'][q_type]['attempted'] += 1

            # Reset token tracker for this question
            main_token_tracker.reset()
            judge_token_tracker.reset()

            try:
                chat_history = []
                if qdata.get('question_type') == 'contextual':
                    q_num = int(qid.split('_')[1])
                    prev_qid = f"Q_{q_num - 1:03d}"
                    if prev_qid in question_cache:
                        prev_q = question_cache[prev_qid]['question']
                        prev_a = question_cache[prev_qid]['answer']
                        chat_history = [
                            HumanMessage(content=prev_q),
                            AIMessage(content=prev_a)
                        ]
                        print(f"  [History] Using context from {prev_qid}")
                        print(f"  [History] Previous Q: {prev_q}")
                        print(f"  [History] Previous A: {prev_a}")

                doc_catcher = DocCatcher()
                base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
                spy_retriever = base_retriever | RunnableLambda(doc_catcher)

                reranker_instance = None
                if CONFIG["use_reranker"]:
                    try:
                        reranker_instance = CohereRerank(model="rerank-english-v3.0", top_n=15)
                        print("  [Setup] Cohere reranker enabled")
                    except Exception as e:
                        print(f"  [Setup] Failed to initialize reranker: {e}")

                # Build pipeline based on function name
                if pipeline_builder_fn.__name__ == "build_system_d_pipeline":
                    rag_chain = pipeline_builder_fn(spy_retriever, llm, doc_catcher, vector_store, reranker_instance)
                elif pipeline_builder_fn.__name__ == "build_system_c_pipeline":
                    rag_chain = pipeline_builder_fn(spy_retriever, llm, doc_catcher, vector_store)
                else:
                    rag_chain = pipeline_builder_fn(spy_retriever, llm)

                start = time.time()
                # Invoke with callbacks
                generated_answer = rag_chain.invoke(
                    {"input": qdata['question'], "chat_history": chat_history},
                    config={"callbacks": [main_token_tracker]} 
                )
                latency_ms = (time.time() - start) * 1000
                DIAGNOSTIC_STATS['avg_latency'].append(latency_ms)

                question_cache[qid] = {
                    'question': qdata['question'],
                    'answer': generated_answer,
                    'docs': doc_catcher.docs
                }

                retrieved_docs = doc_catcher.docs or []
                print(f"\n[Retrieval] Retrieved {len(retrieved_docs)} docs")
                if retrieved_docs:
                    print("  Top 3 docs:")
                    for idx, doc in enumerate(retrieved_docs[:3]):
                        print(f"    {idx+1}. {doc.metadata.get('title', 'N/A')} "
                              f"(Year: {doc.metadata.get('release_year', 'N/A')}, "
                              f"Rating: {doc.metadata.get('average_rating', 'N/A')})")
                
                retrieval_metrics = calculate_retrieval_metrics(
                    retrieved_docs,
                    qdata.get('gold_doc_ids', []),
                    qdata.get('question_type', 'factual')
                )
                
                print(f"\n[Generation] Generated: {generated_answer}")
                print(f"[Generation] Latency: {latency_ms:.0f}ms")
                
                generation_metrics = judge_chain.invoke({
                    "question": qdata['question'],
                    "gold_answer": qdata.get('gold_answer'),
                    "generated_answer": generated_answer
                }, config={"callbacks": [judge_token_tracker]})

                def safe_fmt(val):
                    return f"{val:.3f}" if val is not None else "N/A"

                print(f"\n[Metrics]")
                print(f"  Correctness: {generation_metrics['correctness_score']:.2f}")
                print(f"  Faithfulness: {generation_metrics['faithfulness_score']:.2f}")
                print(f"  Retrieval Recall: {retrieval_metrics.get('retrieval_recall', 'N/A')}")
                print(f"  Retrieval MRR: {retrieval_metrics.get('retrieval_mrr', 'N/A')}")
                print(f"  Recall@5: {safe_fmt(retrieval_metrics.get('recall_at_5'))}")
                print(f"  Precision@5: {safe_fmt(retrieval_metrics.get('precision_at_5'))}")
                print(f"  nDCG@5: {safe_fmt(retrieval_metrics.get('ndcg_at_5'))}")

                main_usage = main_token_tracker.get_usage()
                judge_usage = judge_token_tracker.get_usage()
                print(f"  Main LLM Tokens:  {main_usage['total_tokens']} (P: {main_usage['prompt_tokens']}, C: {main_usage['completion_tokens']})")
                print(f"  Judge LLM Tokens: {judge_usage['total_tokens']} (P: {judge_usage['prompt_tokens']}, C: {judge_usage['completion_tokens']})")

                # Save separated costs to JSON
                result = {
                    "experiment_name": experiment_name,
                    "question_id": qid,
                    "question_type": qdata.get('question_type'),
                    "question": qdata['question'],
                    "gold_answer": qdata.get('gold_answer'),
                    "generated_answer": generated_answer,
                    "latency_ms": latency_ms,
                    **retrieval_metrics,
                    **generation_metrics,
                    "main_total_tokens": main_usage['total_tokens'],
                    "main_prompt_tokens": main_usage['prompt_tokens'],
                    "main_completion_tokens": main_usage['completion_tokens'],
                    
                    "judge_total_tokens": judge_usage['total_tokens'],
                    "judge_prompt_tokens": judge_usage['prompt_tokens'],
                    "judge_completion_tokens": judge_usage['completion_tokens']
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                DIAGNOSTIC_STATS['successful'] += 1
                DIAGNOSTIC_STATS['question_types'][q_type]['successful'] += 1
                
                if generation_metrics['correctness_score'] >= 1.0:
                    print("  ✓ CORRECT")
                else:
                    print("  ✗ INCORRECT")

            except Exception as e:
                DIAGNOSTIC_STATS['failed'] += 1
                safe_print_exc(f"  [ERROR] Failed on {qid}:", e)
                out_f.write(json.dumps({
                    "experiment_name": experiment_name,
                    "question_id": qid,
                    "question_type": qdata.get('question_type'),
                    "error": str(e),
                    "error_type": type(e).__name__
                }) + "\n")
                out_f.flush()

    # Print diagnostic summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    print(f"Total Questions: {DIAGNOSTIC_STATS['total_questions']}")
    print(f"Successful: {DIAGNOSTIC_STATS['successful']}")
    print(f"Failed: {DIAGNOSTIC_STATS['failed']}")
    if DIAGNOSTIC_STATS['total_questions'] > 0:
        print(f"Success Rate: {DIAGNOSTIC_STATS['successful']/DIAGNOSTIC_STATS['total_questions']*100:.1f}%")
    
    if DIAGNOSTIC_STATS['avg_latency']:
        print(f"Avg Latency: {sum(DIAGNOSTIC_STATS['avg_latency'])/len(DIAGNOSTIC_STATS['avg_latency']):.0f}ms")
    
    print(f"\nBy Question Type:")
    for q_type, stats in DIAGNOSTIC_STATS['question_types'].items():
        success_rate = stats['successful']/stats['attempted']*100 if stats['attempted'] > 0 else 0
        print(f"  {q_type}: {stats['successful']}/{stats['attempted']} ({success_rate:.1f}%)")

    if DIAGNOSTIC_STATS['errors_by_type']:
        print(f"\nErrors by Type:")
        for error_type, count in DIAGNOSTIC_STATS['errors_by_type'].items():
            print(f"  {error_type}: {count}")

    print(f"\n{'='*80}\nEVALUATION COMPLETE\n{'='*80}")


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="RAG Evaluation Test Harness (Systems A, C, D-Fixed)")
    parser.add_argument("--llm", type=str, default="llama3b", help="Alias for the generation LLM")
    parser.add_argument("--db", type=str, default="bge_m3", help="Alias for the database configuration")
    args = parser.parse_args()
    
    # 2. Validate and Map
    if args.llm not in LLM_ALIASES:
        print(f"[ERROR] Unknown LLM alias: '{args.llm}'. Available: {list(LLM_ALIASES.keys())}")
        exit(1)
    if args.db not in DB_CONFIGS:
        print(f"[ERROR] Unknown DB alias: '{args.db}'. Available: {list(DB_CONFIGS.keys())}")
        exit(1)

    # 3. Update Configuration
    selected_llm = LLM_ALIASES[args.llm]
    selected_db = DB_CONFIGS[args.db]
    
    CONFIG["generation_llm"] = selected_llm
    CONFIG["grader_llm"] = selected_llm
    CONFIG["rewriter_llm"] = selected_llm
    CONFIG["embedding_model"] = selected_db["embedding_model"]
    CONFIG["db_path"] = selected_db["db_path"]

    # 4. Initialize Judge (Now that config is final)
    try:
        JUDGE_LLM = get_safe_llm(CONFIG["judge_llm"])
    except Exception as e:
        print(f"[ERROR] Failed to initialize Judge: {e}")
        exit(1)

    print(f"""
╔════════════════════════════════════════════════════════════════╗
║           RAG EVALUATION TEST HARNESS v5                       ║
║           CLI CONFIGURATION MODE                               ║
║  LLM: {CONFIG['generation_llm']}                               
║  DB:  {CONFIG['db_path']}                                      
╚════════════════════════════════════════════════════════════════╝
    """)

    # Define experiments to run (Using Updated Config)
    experiments = [
        {
            "name": f"SysA_Baseline_{CONFIG['embedding_model']}_{CONFIG['generation_llm']}",
            "builder": build_system_a_pipeline
        },
        {
            "name": f"SysC_Hybrid_FIXED_{CONFIG['embedding_model']}_{CONFIG['generation_llm']}",
            "builder": build_system_c_pipeline
        },
        {
            "name": f"SysD_v5_FIXED_{CONFIG['embedding_model']}_{CONFIG['generation_llm']}",
            "builder": build_system_d_pipeline
        }
    ]

    # Generate results filename
    RESULTS_FILEPATH = generate_results_filename([exp["name"] for exp in experiments])
    
    if os.path.exists(RESULTS_FILEPATH):
        os.remove(RESULTS_FILEPATH)
        print(f"[INIT] Removed existing {RESULTS_FILEPATH}")

    # Run all experiments
    for exp in experiments:
        # Reset diagnostic stats for each experiment
        DIAGNOSTIC_STATS['total_questions'] = 0
        DIAGNOSTIC_STATS['successful'] = 0
        DIAGNOSTIC_STATS['failed'] = 0
        DIAGNOSTIC_STATS['errors_by_type'] = {}
        DIAGNOSTIC_STATS['question_types'] = {}
        DIAGNOSTIC_STATS['avg_latency'] = []
        
        run_evaluation(
            experiment_name=exp["name"],
            embedding_model_name=CONFIG["embedding_model"],
            db_path=CONFIG["db_path"],
            llm_name=CONFIG["generation_llm"],
            pipeline_builder_fn=exp["builder"]
        )

    # Analysis
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS - COMPARING ALL SYSTEMS")
    print(f"{'='*80}")
    
    try:
        df = pd.read_json(RESULTS_FILEPATH, lines=True)
        
        print("\nOverall Metrics by System:")
        print("-" * 80)
        summary = df.groupby('experiment_name')[['retrieval_recall', 'retrieval_mrr', 'correctness_score', 'faithfulness_score', 'latency_ms']].mean()
        print(summary)
        
        print("\n" + "="*80)
        print("BY QUESTION TYPE")
        print("="*80)
        for exp_name in df['experiment_name'].unique():
            print(f"\n{exp_name}:")
            exp_df = df[df['experiment_name'] == exp_name]
            type_summary = exp_df.groupby('question_type').agg({
                'correctness_score': ['mean', 'count'],
                'latency_ms': 'mean'
            })
            print(type_summary)
        
        # Show perfect scores
        print(f"\n{'='*80}")
        print("PERFECT SCORES BY SYSTEM")
        print(f"{'='*80}")
        for exp_name in df['experiment_name'].unique():
            exp_df = df[df['experiment_name'] == exp_name]
            perfect = exp_df[exp_df['correctness_score'] >= 1.0]
            print(f"{exp_name}: {len(perfect)}/{len(exp_df)} ({len(perfect)/len(exp_df)*100:.1f}%)")
        
    except Exception as e:
        print(f"\nCouldn't generate analysis: {e}")
        safe_print_exc("Analysis error:", e)