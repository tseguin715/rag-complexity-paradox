# OPTIMIZED model_factory.py for Llama3.1-70b-iq2_xs and all Ollama models
# Key improvements:
# 1. Proper configuration for heavily quantized models
# 2. Stop sequences to prevent question-asking
# 3. Reduced context windows for stability
# 4. System prompts for direct answering

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import logging
import os 

# Set up logging for HuggingFace (to suppress non-critical warnings)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- FACTORY 1: EMBEDDING MODELS ---

def get_embedding_model(model_name: str):
    """
    Factory function to load all specified embedding models.
    """
    print(f"[Embedding Factory] Loading model: {model_name}")

    # --- Tier 1: Cloud API ---
    if model_name == "openai-small":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    if model_name == "openai-large":
        return OpenAIEmbeddings(model="text-embedding-3-large")

    # --- Tier 3: Enthusiast (Local High-End GPU) ---
    
    if model_name == "bge-m3":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda'}, 
            encode_kwargs={'normalize_embeddings': True}
        )

    if model_name == "jina-embeddings-v3":
        return HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v3",
            model_kwargs={"device": "cuda", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "task": "retrieval.query"}
        )

    if model_name == "e5-mistral":
        return HuggingFaceEmbeddings(
            model_name="intfloat/e5-mistral-7b-instruct",
            model_kwargs={"device": "cuda", 'trust_remote_code': True},
            encode_kwargs={"normalize_embeddings": True}
        )
        
    # --- Tier 2: Typical (Local, on CPU) ---
    
    if model_name == "minilm":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    raise ValueError(f"Unknown embedding model name: {model_name}")


# --- FACTORY 2: GENERATIVE MODELS ---

def get_generative_model(model_name: str):
    """
    Factory function to load SOTA generative LLMs (Nov 2025 Edition).
    OPTIMIZED FOR QUANTIZED MODELS AND OLLAMA.
    """
    print(f"[LLM Factory] Loading model: {model_name}")
    
    # Common stop sequences for Ollama models
    OLLAMA_STOP_SEQUENCES = [
        "Question:", 
        "User:", 
        "Human:", 
        "\n\n\n",
        "What would you like",
        "What's your question",
        "How can I help",
        "Is there anything"
    ]
    
    # System prompt for direct answering
    DIRECT_ANSWER_SYSTEM = (
        "You are a helpful assistant that answers questions directly based on provided context. "
        "CRITICAL RULES:\n"
        "1. CITATIONS: You must cite your sources using [Doc ID] notation (e.g., 'The movie was released in 2020 [Doc 1].').\n"
        "2. NO OUTSIDE KNOWLEDGE: Answer ONLY using the provided context. If the answer is not in the context, say 'I cannot find the answer in the provided documents'. Do not use your internal training data.\n"
        "3. Never ask questions back to the user.\n"
        "Be concise and specific."
    )

    # --- Tier 0: The New State of the Art (Nov 2025) ---
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=0)

    if model_name == "gpt-5.2":
        return ChatOpenAI(model="gpt-5.2", temperature=0)


    if model_name == "gemini-2.5-pro":
        return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        
    if model_name == "gemini-2.5-flash-lite":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    if model_name == "gemini-2.5-flash":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    if model_name == 'gemini-flash-latest':
        return ChatGoogleGenerativeAI(model='gemini-flash-latest', temperature=0)

    if model_name == "claude-sonnet-4-5":
        return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    if model_name == "claude-3-5-haiku":
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    if model_name == "claude-haiku-4-5":
        return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    # --- Tier 1: Legacy High-End ---
    if model_name == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0)

    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if model_name == "gpt-3.5-turbo":
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # --- OPTIMIZED OLLAMA MODELS ---
    
    # HEAVILY QUANTIZED MODELS (2-bit) - Need special handling
    if model_name == "llama3.1-70b-iq2_xs":
        return ChatOllama(
            model="bazsalanszky/llama3.1:70B-instruct-iq2_xs",
            num_ctx=8192,  # Optimized for stability
            temperature=0,
            num_predict=256,  # Short outputs for quantized model
            top_k=40,  # Conservative token selection
            top_p=0.9,
            repeat_penalty=1.1,
            seed=42,  # Reproducibility
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format="",  # No special formatting
            options={
                "num_gpu": -1,  # Use all available GPUs
                "num_thread": 8,  # Parallel processing
                "low_vram": False,
            }
        )
    
    if model_name == "qwen2-72b-q2_K":
        return ChatOllama(
            model="qwen2.5:72b-instruct-q2_K",
            num_ctx=8192,
            temperature=0,
            num_predict=256,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            seed=42,
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format=""
        )
    
    if model_name == 'qwen2.5:32b-instruct-q2_K':
        return ChatOllama(
            model='qwen2.5:32b-instruct-q2_K',
            num_ctx=8192,
            temperature=0,
            num_predict=256,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            seed=42,
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format=""
        )

    # MODERATELY QUANTIZED / FULL MODELS
    if model_name == "qwen2.5:32b":
        return ChatOllama(
            model="qwen2.5:32b",
            num_ctx=12288,  # Slightly larger for full precision
            temperature=0,
            num_predict=512,  # Can handle longer outputs
            top_k=40,
            top_p=0.95,
            repeat_penalty=1.05,
            seed=42,
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format=""
        )

    if model_name == "llama3.1:70b":
        return ChatOllama(
            model="llama3.1:70b",
            num_ctx=8192,  # Optimized for stability
            temperature=0,
            num_predict=512,
            top_k=10,  # More restrictive for full model
            top_p=0.9,
            repeat_penalty=1.1,
            seed=42,
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format=""
        )
    
    # SMALLER MODELS
    if model_name == "llama3-8b":
        return ChatOllama(
            model="llama3",
            num_ctx=4096,  # Smaller context for 8B model
            temperature=0,
            num_predict=256,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            seed=42,
            stop=OLLAMA_STOP_SEQUENCES,
            system=DIRECT_ANSWER_SYSTEM,
            format=""
        )

    raise ValueError(f"Unknown generative model name: {model_name}")


# --- UTILITY FUNCTIONS FOR QUANTIZED MODELS ---

def get_optimal_system_for_model(model_name: str) -> str:
    """
    Returns the optimal RAG system for a given model.
    """
    recommendations = {
        # Heavily quantized models
        "llama3.1-70b-iq2_xs": "SysD",
        "qwen2-72b-q2_K": "SysD",
        "qwen2.5:32b-instruct-q2_K": "SysA",  # Simpler is better for 2-bit
        
        # Full/moderately quantized models
        "llama3.1:70b": "SysD",
        "qwen2.5:32b": "SysA",
        "llama3-8b": "SysA",  # Small model needs simple approach
        
        # High-end models
        "gpt-4o": "SysG",  # Can handle complexity
        "gpt-5.1": "SysG",
        "claude-sonnet-4-5": "SysG",
        
        # Default
        "default": "SysA"
    }
    
    return recommendations.get(model_name, recommendations["default"])


def get_model_specific_prompt_adjustments(model_name: str, base_prompt: str) -> str:
    """
    Adjusts prompts for specific model quirks and limitations.
    """
    if "iq2_xs" in model_name or "q2_K" in model_name:
        # Heavily quantized models need simpler prompts
        return f"""Answer the question using the context.
Be direct and specific.

{base_prompt}

Answer:"""
    
    elif "llama" in model_name.lower():
        # Llama models benefit from explicit structure
        return f"""{base_prompt}

Provide a direct answer based on the context above.
Do not ask questions or request clarification.

Answer:"""
    
    else:
        # Default - return as-is
        return base_prompt


# --- CONFIGURATION VALIDATOR ---

def validate_ollama_config(model_name: str):
    """
    Validates and warns about Ollama model configurations.
    """
    warnings = []
    
    model = get_generative_model(model_name)
    
    if hasattr(model, 'num_ctx') and model.num_ctx > 12288:
        warnings.append(f"⚠️  {model_name}: num_ctx={model.num_ctx} may be too large")
    
    if not hasattr(model, 'stop') or not model.stop:
        warnings.append(f"⚠️  {model_name}: Missing stop sequences")
    
    if not hasattr(model, 'system') or not model.system:
        warnings.append(f"⚠️  {model_name}: Missing system prompt")
    
    if hasattr(model, 'temperature') and model.temperature > 0:
        warnings.append(f"⚠️  {model_name}: temperature={model.temperature} should be 0")
    
    if warnings:
        print(f"\nConfiguration warnings for {model_name}:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print(f"✅ {model_name} configuration validated")
    
    return len(warnings) == 0


if __name__ == "__main__":
    # Test configuration validation
    print("="*60)
    print("VALIDATING OLLAMA CONFIGURATIONS")
    print("="*60)
    
    ollama_models = [
        "llama3.1-70b-iq2_xs",
        "llama3.1:70b",
        "qwen2.5:32b",
        "llama3-8b"
    ]
    
    for model in ollama_models:
        try:
            validate_ollama_config(model)
        except Exception as e:
            print(f"❌ {model}: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDED SYSTEMS BY MODEL")
    print("="*60)
    
    for model in ollama_models:
        system = get_optimal_system_for_model(model)
        print(f"{model:30} → {system}")