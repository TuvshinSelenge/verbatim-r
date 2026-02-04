"""
RAG Pipeline Service - Consolidated workflow with BGE reranker.

This service provides an alternative RAG pipeline that uses:
1. Query rewriting and multi-query generation
2. Milvus hybrid search (dense + sparse)
3. BGE reranker for improved ranking
4. LLM span extraction for verbatim answers
"""

import os
import asyncio
import json
import re
from typing import AsyncGenerator, Dict, Any, List, Optional
from types import SimpleNamespace
from dataclasses import dataclass

from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymilvus import connections
from openai import OpenAI
import openai

from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.embedding_providers import SpladeProvider
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import TemplateManager
from verbatim_rag.llm_client import LLMClient
from verbatim_rag.models import DocumentWithHighlights
from sentence_transformers import SentenceTransformer

from custom.query_rewriter import QueryRewriter
from custom.query_generator import QueryGenerator
from custom.bge_ranker import BGEReranker

# To run the RAG
# use this : PYTHONPATH=. uvicorn api.app:app --reload --port 8000

# Load in different terminal the frontend with npm run dev


load_dotenv()

DB_PATH = os.getenv("CUSTOM_DB_PATH", "./custom/milvus_verbatim.db")

BANK_NAME = "Raiffeisen Bank International AG"
BANK_SHORT = "RBI"

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LLM_MODEL = "google/gemini-3-flash-preview"


# Robust JSON parsing for Gemini/OpenRouter responses

def safe_parse_json(response: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks and extra text.
    Gemini often wraps JSON in ```json ... ``` blocks.
    """
    if not response or not response.strip():
        raise ValueError("Empty response")
    
    content = response.strip()

    # Method 1: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Method 2: Find the first { and last } (handles text before/after JSON)
    brace_match = re.search(r'(\{[\s\S]*\})', content)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Method 3: Direct parse (already valid JSON)
    return json.loads(content)


# Embedding Provider

class LocalHuggingFaceProvider:
    """Dense embedding provider using local HuggingFace models."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        print(f"🔹 Loading Dense Embedder: {model_name} on {device}...")
        # Load on CPU first, then move to target device
        self.model = SentenceTransformer(model_name) 
        self.device = device
        self.model = self.model.to(device)
        
    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)


# RAG Service

class RAG:
    """
    Complete RAG pipeline service.
    
    This service orchestrates:
    1. Multi-query generation
    2. Hybrid search (dense + sparse)
    3. BGE reranking
    4. LLM span extraction
    5. Response generation
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("Initializing RAG Service...")

        self.bank_name = BANK_NAME
        self.bank_short = BANK_SHORT
        
        self.openai_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        )   
        
        # Initialize components
        self._init_embedders()
        self._init_vector_store()
        self._init_reranker()
        self._init_query_rewriter()
        self._init_query_generator()
        self._init_extractor()
        
        self._initialized = True
        print("RAG Service ready!")
    
    def _init_embedders(self):
        """Initialize embedding providers."""
        device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dense_embedder = LocalHuggingFaceProvider(EMBEDDING_MODEL, device)
        self.sparse_provider = SpladeProvider(model_name=SPARSE_MODEL, device="cpu")
        print("Embedders initialized")
    
    def _init_vector_store(self):
        """Initialize vector store and index."""
        print(f"Connecting to database: {DB_PATH}")
        
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Database not found at {DB_PATH}\n"
                "Please ensure the database exists."
            )
        
        # Disconnect if already connected
        try:
            connections.disconnect("default")
        except Exception:
            pass
        
        self.store = LocalMilvusStore(
            DB_PATH,
            enable_sparse=True,
            enable_dense=True,
        )
        
        self.index = VerbatimIndex(
            vector_store=self.store,
            dense_provider=self.dense_embedder,
            sparse_provider=self.sparse_provider,
        )
        print("Vector store connected")
    
    def _init_reranker(self):
        """Initialize BGE reranker."""
        self.reranker = BGEReranker()
    
    def _init_query_rewriter(self):
        """Initialize query rewriter."""
        self.query_rewriter = QueryRewriter(openai_client=self.openai_client, model=LLM_MODEL)
        print("Query rewriter initialized")
    
    def _init_query_generator(self):
        """Initialize query generator."""
        self.query_generator = QueryGenerator(self.openai_client, LLM_MODEL)
    
    def _init_extractor(self):
        """Initialize LLM span extractor and response builder."""
        self.llm_client = LLMClient(
            model=LLM_MODEL,
            api_base=OPENROUTER_BASE_URL,
        )
        self.llm_client.client = self.openai_client
        self.llm_client.async_client = openai.AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        # Patch LLMClient for robust JSON parsing (handles Gemini's markdown-wrapped responses)
        self._patch_llm_client_for_robust_json()

        self.extractor = LLMSpanExtractor(llm_client=self.llm_client)
        self.response_builder = ResponseBuilder()
        self.template_manager = TemplateManager(llm_client=self.llm_client)
        self.template_manager.use_contextual_mode(use_per_fact=True) # Added to activate contextual mode
        print("Template manager initialized")
    
    def _patch_llm_client_for_robust_json(self):
        """
        Patch the LLMClient to handle various JSON response formats from Gemini.
        Gemini often returns JSON wrapped in markdown code blocks.
        """
        original_complete = self.llm_client.complete
        original_complete_async = self.llm_client.complete_async
        llm_client = self.llm_client
        
        def robust_extract_spans(question: str, documents: dict) -> dict:
            """Extract spans with robust JSON parsing."""
            prompt = llm_client._build_extraction_prompt(question, documents)
            try:
                response = original_complete(prompt, json_mode=True)
                return safe_parse_json(response)
            except (json.JSONDecodeError, ValueError) as e:
                # If json_mode fails, try without it
                try:
                    response = original_complete(prompt, json_mode=False)
                    return safe_parse_json(response)
                except (json.JSONDecodeError, ValueError) as e2:
                    print(f"Span extraction failed: {e2}")
                    if response:
                        print(f"Raw response preview: {response[:300]}...")
                    return {doc_id: [] for doc_id in documents.keys()}
        
        async def robust_extract_spans_async(question: str, documents: dict) -> dict:
            """Async extract spans with robust JSON parsing."""
            prompt = llm_client._build_extraction_prompt(question, documents)
            try:
                response = await original_complete_async(prompt, json_mode=True)
                return safe_parse_json(response)
            except (json.JSONDecodeError, ValueError) as e:
                try:
                    response = await original_complete_async(prompt, json_mode=False)
                    return safe_parse_json(response)
                except (json.JSONDecodeError, ValueError) as e2:
                    print(f"Async span extraction failed: {e2}")
                    if response:
                        print(f"Raw response preview: {response[:300]}...")
                    return {doc_id: [] for doc_id in documents.keys()}
        
        # Patch the methods
        self.llm_client.extract_spans = robust_extract_spans
        self.llm_client.extract_spans_async = robust_extract_spans_async
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._initialized and self.index is not None
    
    async def stream_query(
        self,
        question: str,
        num_docs: int = 5,
        per_query_k: int = 20,
        bank_name: str = BANK_NAME,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a query response using the rag pipeline.
        
        Pipeline:
        1. Generate multiple search queries
        2. Search for each query and merge results
        3. Rerank with BGE
        4. Extract spans
        5. Generate answer
        
        Yields:
            Dictionary with type and data for each stage
        """
        try:
            # Step 0: Rewrite the question for better search
            print(f"Rewriting query: {question[:50]}...")
            rewritten_question = await asyncio.to_thread(
                self.query_rewriter.rewrite, question
            )
            # Use the rewritten query for downstream LLM span extraction (keeps prompts aligned)
            span_extraction_question = (
            f"{rewritten_question}\n\n"
            f"""Clarification: If a question mentions 'contracting party',
             'organisation', 'entity', 'firm', 'group' then it refers to {self.bank_name})."""
            )   

            # Step 1: Generate search queries from the rewritten question
            print(f"Generating queries...")
            queries = await asyncio.to_thread(
                self.query_generator.generate_queries, rewritten_question
            )
            print(f"Generated {len(queries)} queries")
            
            # Step 2: Search for each query and merge results
            print("Searching...")
            merged = []
            seen = set()
            
            for q in queries:
                hits = await asyncio.to_thread(self.index.query, q, per_query_k)
                for h in hits:
                    text = self._get_text(h)
                    meta = self._get_meta(h)
                    text = self._normalize_text(text)
                    if not text:
                        continue
                    
                    # Create dedup key
                    key = (
                        meta.get("source_file") or meta.get("document_id") or "",
                        meta.get("chunk_index"),
                        text[:200]
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(h)
            
            print(f"Found {len(merged)} unique chunks")
            
            # Send documents without highlights first
            documents_without_highlights = []
            for doc in merged[:num_docs]:  # Send only requested number
                meta = self._get_meta(doc)
                title, source = self._extract_title_and_source(meta)
                documents_without_highlights.append(
                    DocumentWithHighlights(
                        content=self._get_text(doc),
                        highlights=[],
                        title=title,
                        source=source,
                        metadata=meta,
                    )
                )
            
            yield {
                "type": "documents",
                "data": [doc.model_dump() for doc in documents_without_highlights],
            }
            
            # Step 3: Rerank with BGE
            print("Reranking...")
            top_chunks, ranking = await asyncio.to_thread(
                # Notebook uses the rewritten question for reranking
                self.reranker.rerank, rewritten_question, merged, num_docs
            )
            print(f"Top {len(top_chunks)} chunks after reranking")
            
            # Wrap chunks for extractor
            wrapped_chunks = []
            for c in top_chunks:
                text = self._get_text(c)
                meta = self._get_meta(c)
                wrapped_chunks.append(
                    SimpleNamespace(
                        text=self._normalize_text(text),
                        metadata=meta,
                        source_file=meta.get("source_file"),
                        page=meta.get("page"),
                    )
                )
            
            # Update documents with reranked order
            reranked_documents = []
            for chunk in wrapped_chunks:
                title, source = self._extract_title_and_source(chunk.metadata)
                reranked_documents.append(
                    DocumentWithHighlights(
                        content=chunk.text,
                        highlights=[],
                        title=title,
                        source=source,
                        metadata=chunk.metadata,
                    )
                )
            
            yield {
                "type": "documents",
                "data": [doc.model_dump() for doc in reranked_documents],
            }
            
            # Step 4: Extract spans
            print("Extracting spans...")
            relevant_spans = await asyncio.to_thread(
                # Use the rewritten query for LLM-driven span extraction
                self.extractor.extract_spans, span_extraction_question, wrapped_chunks
            )
            
            # Create highlights
            interim_documents = []
            for chunk in wrapped_chunks:
                doc_content = chunk.text
                doc_spans = relevant_spans.get(doc_content, [])
                if doc_spans:
                    highlights = self.response_builder._create_highlights(doc_content, doc_spans)
                else:
                    highlights = []
                title, source = self._extract_title_and_source(chunk.metadata)
                interim_documents.append(
                    DocumentWithHighlights(
                        content=doc_content,
                        highlights=highlights,
                        title=title,
                        source=source,
                        metadata=chunk.metadata,
                    )
                )
            
            yield {
                "type": "highlights",
                "data": [d.model_dump() for d in interim_documents],
            }
            
            # Step 5: Generate answer
            print("Generating answer...")
            
            # Rank spans and split into display vs citation-only
            display_spans, citation_spans = self._rank_and_split_spans(relevant_spans)
            
            # Generate answer using template manager
            try:
                answer = await self.template_manager.process_async(
                    question, display_spans, citation_spans
                )
                answer = self.response_builder.clean_answer(answer)
            except Exception as e:
                print(f"Template processing failed: {e}")
                yield {
                    "type": "error",
                    "error": f"template_processing_failed: {e}",
                    "done": True,
                }
                return
            
            # Build final response
            result = self.response_builder.build_response(
                question=question,
                answer=answer,
                search_results=wrapped_chunks,
                relevant_spans=relevant_spans,
                display_span_count=len(display_spans),
            )
            
            yield {"type": "answer", "data": result.model_dump(), "done": True}
            
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            import traceback
            traceback.print_exc()
            yield {"type": "error", "error": str(e), "done": True}
    
    def _rank_and_split_spans(
        self, 
        relevant_spans: Dict[str, List[str]], 
        max_display: int = 5
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Rank all spans and split into display spans and citation-only spans.
        
        Returns:
            Tuple of (display_spans, citation_spans) as lists of dicts with "text" key
        """
        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span})
        
        # Take top max_display for display, rest are citations
        display_spans = all_spans[:max_display]
        citation_spans = all_spans[max_display:]
        
        return display_spans, citation_spans
    
    @staticmethod
    def _get_text(c) -> str:
        if isinstance(c, dict):
            return c.get("text", "")
        return getattr(c, "text", "")
    
    @staticmethod
    def _get_meta(c) -> dict:
        if isinstance(c, dict):
            return c.get("metadata", {}) or {}
        return getattr(c, "metadata", {}) or {}
    
    @staticmethod
    def _normalize_text(t: str) -> str:
        return (t or "").replace("\ufffd", "").replace("\r", "\n")
    
    @staticmethod
    def _extract_title_and_source(meta: dict) -> tuple[str, str]:
        """
        Extract a meaningful title and source from chunk metadata.
        
        Tries various common field names and builds a descriptive title
        including page number if available.
        """
    
        source = (
            meta.get("source") or 
            meta.get("source_file") or 
            meta.get("filename") or 
            meta.get("file_path") or
            ""
        )
        
        # Extract just the filename from path if it's a full path
        if source and "/" in source:
            source = source.split("/")[-1]
        
        # Try to get title
        title = meta.get("title") or ""
        
        # If no title, use source filename as title
        if not title and source:
            # Remove extension for cleaner title
            title = source.rsplit(".", 1)[0] if "." in source else source
        
        # Add page number to title if available
        page = meta.get("page_number") or meta.get("page") or meta.get("page_num")
        if page is not None:
            if title:
                title = f"{title} (Page {page})"
            else:
                title = f"Page {page}"
        
        # Add chunk number if available and no page
        if not page:
            chunk_num = meta.get("chunk_number") or meta.get("chunk_index")
            if chunk_num is not None and title:
                title = f"{title} (Chunk {chunk_num})"
        
        # Fallback to document ID if still no title
        if not title:
            doc_id = meta.get("document_id") or meta.get("doc_id")
            if doc_id:
                title = f"Document {doc_id[:8]}..."
            else:
                title = "Document"
        
        return title, source


# Singleton accessor
_service_instance: Optional[RAG] = None


def get_custom_service() -> RAG:
    """Get or create the Custom RAG service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RAG()
    return _service_instance



