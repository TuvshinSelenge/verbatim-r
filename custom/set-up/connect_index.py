"""
Milvus Index Connection Module
==============================
Safely connects to an EXISTING Milvus database without recreating it.

Use this module to connect to your index for querying.
For creating/recreating the index, use create_index.py instead.

Usage:
    from connect_index import connect_to_index, get_embedders
    
    rag_index, client = connect_to_index()
"""

import os
from typing import Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_DB_PATH = os.getenv("DB_PATH", "./milvus_verbatim.db")


def get_embedders(device: str = "cpu") -> Tuple[Any, Any]:
    """
    Create dense and sparse embedding providers.
    
    Args:
        device: Device to use (cpu/cuda)
        
    Returns:
        Tuple of (dense_embedder, sparse_provider)
    """
    import verbatim_rag.embedding_providers as ep
    from verbatim_rag.embedding_providers import SpladeProvider
    from sentence_transformers import SentenceTransformer
    
    BaseClass = ep.SpladeProvider.__bases__[0] if ep.SpladeProvider.__bases__ else object
    
    class LocalHuggingFaceProvider(BaseClass):
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
            print(f"🔹 Loading Model: {model_name} on {device}...")
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
            
        def get_dimension(self):
            return self.model.get_sentence_embedding_dimension()

        def embed_text(self, text):
            return self.model.encode(text, normalize_embeddings=True).tolist()
            
        def embed_batch(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        def embed_documents(self, texts): 
            return self.embed_batch(texts)
        
        def embed_query(self, text): 
            return self.embed_text(text)
    
    dense_embedder = LocalHuggingFaceProvider(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=device
    )
    
    return dense_embedder, sparse_provider


def connect_to_index(
    db_path: str = DEFAULT_DB_PATH,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Connect to an existing Milvus index.
    
    This function safely connects to an existing database WITHOUT
    deleting or modifying it. Use create_index.py to create/recreate.
    
    Args:
        db_path: Path to the Milvus database
        device: Device for embeddings (cpu/cuda)
        verbose: Print progress messages
        
    Returns:
        Tuple of (rag_index, openai_client)
        
    Raises:
        FileNotFoundError: If database doesn't exist
    """
    from pymilvus import connections
    from verbatim_rag.vector_stores import LocalMilvusStore
    from verbatim_rag.index import VerbatimIndex
    from openai import OpenAI
    
    if verbose:
        print("🔌 Connecting to Milvus index...")
    
    # Check if database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"❌ Database not found at {db_path}\n"
            "   Run 'python create_index.py' to create the database first."
        )
    
    # Disconnect any existing connections
    try:
        connections.disconnect("default")
    except Exception:
        pass
    
    # Load embedders FIRST to get the dimension
    if verbose:
        print("📦 Loading embedding models...")
    dense_embedder, sparse_provider = get_embedders(device=device)
    vector_dim = dense_embedder.get_dimension()
    if verbose:
        print(f"   Dense dimension: {vector_dim}")
    
    # Connect to vector store with the correct dimension
    try:
        # IMPORTANT: Pass the correct dense_dim to match the embedding model
        # BAAI/bge-m3 produces 1024-dimensional embeddings
        store = LocalMilvusStore(
            db_path, 
            enable_sparse=True, 
            enable_dense=True,
            dense_dim=vector_dim  # Must match embedding model dimension!
        )
        if verbose:
            print("✅ Connected to vector store")
    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to vector store: {e}")
    
    # Note: embedders were already loaded above to get dimension
    
    # Create index
    try:
        rag_index = VerbatimIndex(
            vector_store=store,
            dense_provider=dense_embedder,
            sparse_provider=sparse_provider,
        )
        if verbose:
            print("✅ Query engine ready")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to create query engine: {e}")
    
    # Create OpenAI client
    client = OpenAI()
    
    if verbose:
        print(f"✅ Connected to: {db_path}")
    
    return rag_index, client


# Quick access function for notebooks
def quick_connect(verbose: bool = True) -> Tuple[Any, Any]:
    """
    Quick connection to the default database.
    
    Usage in notebook:
        from connect_index import quick_connect
        rag_index, client = quick_connect()
    """
    return connect_to_index(verbose=verbose)


if __name__ == "__main__":
    print("=" * 60)
    print("🔌 MILVUS INDEX CONNECTION")
    print("=" * 60)
    
    try:
        rag_index, client = connect_to_index()
        print("\n✅ Successfully connected!")
        print("   You can now use rag_index for queries.")
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")

