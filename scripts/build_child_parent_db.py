
import json
import os
import shutil
import time
from typing import List, Dict
from tqdm import tqdm # You might need to pip install tqdm

# LangChain Imports
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import your factory
from model_factory import get_embedding_model

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

SOURCE_FILE = "source_data.ndjson"

# We will apply this technique to all your targets
TARGET_DATABASES = [
    {
        "model_name": "bge-m3",
        "db_path": "db/chroma_db_bge_m3" 
    },
    {
        "model_name": "jina-embeddings-v3",
        "db_path": "db/chroma_db_jina_v3"
    },
    {
        "model_name": "openai-large",
        "db_path": "db/chroma_db_openai_large"
    },
    {
        "model_name": "openai-small",
        "db_path": "db/chroma_db_openai_small"
    }
]

# ==============================================================================
#  PARENT-CHILD LOGIC
# ==============================================================================

def prepare_batch_for_ingestion(items: List[Dict], embedding_model):
    """
    Creates multiple 'Child' embeddings that all point to the 'Parent' text.
    """
    
    # 1. Prepare lists for manual embedding generation
    texts_to_embed = []
    
    # 2. Prepare lists for storage (The Parent content)
    docs_to_store = []
    metadatas_to_store = []
    ids_to_store = []
    
    for item in items:
        # --- A. CONSTRUCT PARENT TEXT (What the LLM reads) ---
        # This is the rich context we want to return regardless of which vector matched
        parent_text = (
            f"Title: {item.get('title', 'Unknown')}\n"
            f"Year: {item.get('year', 'N/A')}\n"
            f"Type: {item.get('type', 'Movie')}\n"
            f"Genres: {item.get('genres', 'N/A')}\n"
            f"Rating: {item.get('rating', 'N/A')}\n"
            f"Description: {item.get('description', '')}"
        )
        
        # --- B. PREPARE METADATA (Shared by all children) ---
        # Crucial: Use standard keys expected by main_cli.py
        base_metadata = {
            "id": item.get("id"),
            "title": item.get("title"),
            "type": item.get("type"),
            "release_year": item.get("year"),
            "average_rating": item.get("rating"),
            "original_language": item.get("language"),
            "popularity": item.get("popularity"),
            "genre": str(item.get("genres", [])),
        }
        # Clean None values
        base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

        doc_id = str(item.get("id"))
        
        # --- C. CREATE CHILD CHUNKS (The Vectors) ---
        
        # Child 1: The Title (Fixes exact match failures)
        title_text = f"{item.get('title', '')}"
        if title_text:
            texts_to_embed.append(title_text)
            docs_to_store.append(parent_text) # Store full parent
            
            meta = base_metadata.copy()
            meta["chunk_type"] = "title_match"
            metadatas_to_store.append(meta)
            
            ids_to_store.append(f"{doc_id}-title")

        # Child 2: The Description (Standard semantic search)
        desc_text = item.get('description', '')
        if desc_text:
            texts_to_embed.append(desc_text)
            docs_to_store.append(parent_text) # Store full parent
            
            meta = base_metadata.copy()
            meta["chunk_type"] = "description_match"
            metadatas_to_store.append(meta)
            
            ids_to_store.append(f"{doc_id}-desc")
            
        # Child 3: Metadata Soup (Fixes "Horror movies from 1980" queries)
        # We create a dense sentence describing the item
        meta_text = f"{item.get('title')} is a {item.get('genres')} {item.get('type')} released in {item.get('year')}."
        texts_to_embed.append(meta_text)
        docs_to_store.append(parent_text)
        
        meta = base_metadata.copy()
        meta["chunk_type"] = "metadata_match"
        metadatas_to_store.append(meta)
        
        ids_to_store.append(f"{doc_id}-meta")

    # 3. Generate Embeddings (Batch)
    # This is efficient because we send one large list to the GPU/API
    if texts_to_embed:
        embeddings = embedding_model.embed_documents(texts_to_embed)
    else:
        embeddings = []
        
    return ids_to_store, embeddings, docs_to_store, metadatas_to_store

# ==============================================================================
#  BUILDER
# ==============================================================================

def build_database(config: Dict[str, str], raw_data: List[Dict]):
    model_name = config['model_name']
    db_path = config['db_path']
    
    print(f"\n{'='*60}")
    print(f"BUILDING: {model_name} (Parent-Child Mode)")
    print(f"PATH:     {db_path}")
    print(f"{'='*60}")
    
    # 1. Clean existing DB
    if os.path.exists(db_path):
        print(f"[Setup] Removing existing database...")
        try:
            shutil.rmtree(db_path)
            time.sleep(1)
        except Exception as e:
            print(f"[Error] Could not delete {db_path}: {e}")
            return

    # 2. Initialize Model
    try:
        embedding_model = get_embedding_model(model_name)
        
        # Optimization for Jina v3
        if "jina" in model_name.lower():
            if hasattr(embedding_model, 'encode_kwargs'):
                embedding_model.encode_kwargs['task'] = 'retrieval.passage'
            else:
                embedding_model.encode_kwargs = {'task': 'retrieval.passage'}
    except Exception as e:
        print(f"[Error] Model init failed: {e}")
        return

    # 3. Initialize Chroma (Empty)
    vector_store = Chroma(
        collection_name="langchain",
        embedding_function=embedding_model, # Only used if we add_documents (we won't)
        persist_directory=db_path
    )
    
    # 4. Process in Batches
    # We process raw items in batches of 50 (which becomes ~150 vectors)
    BATCH_SIZE = 50 
    total_items = len(raw_data)
    
    print(f"[Ingest] Processing {total_items} source items...")
    
    for i in tqdm(range(0, total_items, BATCH_SIZE)):
        batch_items = raw_data[i : i + BATCH_SIZE]
        
        # Generate the Parent-Child data
        ids, embeddings, documents, metadatas = prepare_batch_for_ingestion(
            batch_items, 
            embedding_model
        )
        
        # Direct Injection into Chroma
        # We bypass LangChain's embedding calculation because we already did it
        # on the specific Child chunks.
        if ids:
            vector_store._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents # This is the Parent Text
            )
            
    print(f"[Success] Database built with {vector_store._collection.count()} vectors.")

# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        exit(1)
        
    # Load Raw JSON
    raw_data = []
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    raw_data.append(json.loads(line))
                except:
                    continue
    
    print(f"Loaded {len(raw_data)} raw items.")

    for config in TARGET_DATABASES:
        build_database(config, raw_data)