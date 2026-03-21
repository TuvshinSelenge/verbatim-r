"""
Milvus Index Connection Module
Safely connects to an existing Milvus database without recreating it.
"""

import os
from typing import Tuple, Any
from dotenv import load_dotenv

load_dotenv()

from custom.setup.paths import resolve_custom_milvus_db_path

# Same resolution as custom RAG (CUSTOM_DB_PATH / DB_PATH / new.db if present / verbatim.db).
DEFAULT_DB_PATH = resolve_custom_milvus_db_path()


def get_embedders(device: str = "cpu") -> Tuple[Any, Any]:
    """Create dense and sparse embedding providers."""
    import verbatim_rag.embedding_providers as ep
    from verbatim_rag.embedding_providers import SpladeProvider
    from sentence_transformers import SentenceTransformer

    BaseClass = ep.SpladeProvider.__bases__[0] if ep.SpladeProvider.__bases__ else object

    class LocalHuggingFaceProvider(BaseClass):
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
            print(f"Loading model: {model_name} on {device}...")
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

    dense_embedder = LocalHuggingFaceProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=device,
    )
    return dense_embedder, sparse_provider


def connect_to_index(
    db_path: str = DEFAULT_DB_PATH,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """Connect to an existing Milvus index."""
    from pymilvus import connections
    from verbatim_rag.vector_stores import LocalMilvusStore
    from verbatim_rag.index import VerbatimIndex
    from openai import OpenAI

    if verbose:
        print("Connecting to Milvus index...")

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}\n"
            "Run 'python create_index.py' to create the database first."
        )

    try:
        connections.disconnect("default")
    except Exception:
        pass

    dense_embedder, sparse_provider = get_embedders(device=device)
    vector_dim = dense_embedder.get_dimension()

    store = LocalMilvusStore(
        db_path,
        enable_sparse=True,
        enable_dense=True,
        dense_dim=vector_dim,
    )

    rag_index = VerbatimIndex(
        vector_store=store,
        dense_provider=dense_embedder,
        sparse_provider=sparse_provider,
    )

    client = OpenAI()
    if verbose:
        print(f"Connected to: {db_path}")
    return rag_index, client


def quick_connect(verbose: bool = True) -> Tuple[Any, Any]:
    """Quick connection to the default database."""
    return connect_to_index(verbose=verbose)
