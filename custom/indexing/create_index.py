"""
Create or incrementally update a local Milvus index from annual-report PDFs.

The script uses Docling for PDF parsing and structure-aware chunking, including
Markdown table serialization. Dense and sparse vectors are generated with the
same providers used by the custom retrieval pipeline.

Examples:
    python -m custom.indexing.create_index
    python -m custom.indexing.create_index \
        --pdf-folder /path/to/pdfs \
        --db-path custom/milvus_verbatim_new.db
    python -m custom.indexing.create_index --force
    python -m custom.indexing.create_index --incremental
    python -m custom.indexing.create_index \
        --incremental --reindex "Report A.pdf" "Report B.pdf"
    python -m custom.indexing.create_index --incremental --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

CUSTOM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION_NAME = "verbatim_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPARSE_MODEL = (
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
)
DEFAULT_PDF_FOLDER = Path(
    os.getenv("PDF_SOURCE_FOLDER", str(CUSTOM_ROOT / "pdfs"))
).expanduser()
DEFAULT_DB_PATH = Path(
    os.getenv(
        "CUSTOM_DB_PATH",
        os.getenv("DB_PATH", str(CUSTOM_ROOT / "milvus_verbatim_new.db")),
    )
).expanduser()


@dataclass
class ProcessedChunk:
    """A Docling chunk and the metadata stored alongside it."""

    text: str
    source_file: str
    page: Optional[int]
    chunk_index: int
    headings: list[str]
    chunk_type: str
    token_count: int


def milvus_filter_string(value: str) -> str:
    """Escape a string for use inside a double-quoted Milvus filter literal."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def infer_vector_store_settings(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[int, bool, bool]:
    """Read dense and sparse configuration from an existing collection."""

    desc = client.describe_collection(collection_name=collection_name)
    dense_dim = 384
    enable_dense = False
    enable_sparse = False

    for field in desc.get("fields", []):
        name = field.get("name")
        if name == "dense_vector":
            enable_dense = True
            dense_dim = int(field.get("params", {}).get("dim", 384))
        elif name == "sparse_vector":
            enable_sparse = True

    if not enable_dense and not enable_sparse:
        raise ValueError(
            f"Collection {collection_name!r} has no dense or sparse vector field."
        )

    return dense_dim, enable_dense, enable_sparse


def pdf_has_indexed_chunks(store: Any, pdf_basename: str) -> bool:
    """Return whether the index already contains chunks from a PDF."""

    filt = (
        f'metadata["source_file"] == '
        f'"{milvus_filter_string(pdf_basename)}"'
    )
    rows = store.client.query(
        collection_name=store.collection_name,
        filter=filt,
        output_fields=["id"],
        limit=1,
    )
    return bool(rows)


def delete_chunks_for_pdf(
    store: Any,
    pdf_basename: str,
    batch_size: int = 512,
) -> int:
    """Delete all chunk and document records associated with a PDF."""

    client = store.client
    chunks_collection = store.collection_name
    documents_collection = store.documents_collection_name
    filt = (
        f'metadata["source_file"] == '
        f'"{milvus_filter_string(pdf_basename)}"'
    )
    removed = 0

    while True:
        rows = client.query(
            collection_name=chunks_collection,
            filter=filt,
            output_fields=["id"],
            limit=batch_size,
        )
        if not rows:
            break

        ids = [row["id"] for row in rows if row.get("id")]
        if not ids:
            break

        store.delete(ids)
        try:
            client.delete(
                collection_name=documents_collection,
                filter=filt,
            )
        except Exception:
            document_rows = client.query(
                collection_name=documents_collection,
                filter=filt,
                output_fields=["id"],
                limit=batch_size,
            )
            document_ids = [
                row["id"] for row in document_rows if row.get("id")
            ]
            if document_ids:
                quoted_ids = ",".join(
                    f'"{milvus_filter_string(str(document_id))}"'
                    for document_id in document_ids
                )
                client.delete(
                    collection_name=documents_collection,
                    filter=f"id in [{quoted_ids}]",
                )

        removed += len(ids)

    return removed


def confirm_deletion(db_path: Path, assume_yes: bool = False) -> bool:
    """Ask before deleting an existing database."""

    if not db_path.exists() or assume_yes:
        return True

    print("\nWARNING: this will delete the existing database:")
    print(f"  {db_path}")
    response = input("Continue? (yes/no): ").strip().lower()
    return response in {"yes", "y"}


def create_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> Any:
    """Create the normalized dense embedding provider."""

    import verbatim_rag.embedding_providers as embedding_providers
    from sentence_transformers import SentenceTransformer

    base_class = (
        embedding_providers.SpladeProvider.__bases__[0]
        if embedding_providers.SpladeProvider.__bases__
        else object
    )

    class LocalHuggingFaceProvider(base_class):
        def __init__(self, selected_model: str, selected_device: str) -> None:
            print(f"Loading dense model {selected_model} on {selected_device}...")
            self.model = SentenceTransformer(
                selected_model,
                device=selected_device,
                trust_remote_code=True,
            )

        def get_dimension(self) -> int:
            return self.model.get_sentence_embedding_dimension()

        def embed_text(self, text: str) -> list[float]:
            return self.model.encode(text, normalize_embeddings=True).tolist()

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
            ).tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_batch(texts)

        def embed_query(self, text: str) -> list[float]:
            return self.embed_text(text)

    return LocalHuggingFaceProvider(model_name, device)


class DoclingProcessor:
    """Parse and chunk PDF documents with Docling."""

    def __init__(
        self,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
        max_tokens: int = 512,
        verbose: bool = True,
    ) -> None:
        self.embedding_model_id = embedding_model_id
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._initialize_converter()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _initialize_converter(self) -> None:
        """Configure Docling PDF parsing and table recognition."""

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        self._log("Initializing Docling document converter...")
        pdf_options = PdfPipelineOptions()
        pdf_options.do_table_structure = True
        pdf_options.do_ocr = False
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )

    def _create_chunker(self) -> Any:
        """Create a HybridChunker with Markdown table serialization."""

        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from docling_core.transforms.serializer.markdown import (
            MarkdownTableSerializer,
        )
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.embedding_model_id),
            max_tokens=self.max_tokens,
        )

        class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
            def get_serializer(self, doc: Any) -> Any:
                return ChunkingDocSerializer(
                    doc=doc,
                    table_serializer=MarkdownTableSerializer(),
                )

        return HybridChunker(
            tokenizer=tokenizer,
            serializer_provider=MarkdownTableSerializerProvider(),
            merge_peers=True,
        )

    def process_pdf(self, pdf_path: Path) -> list[ProcessedChunk]:
        """Convert and chunk one PDF file."""

        from docling_core.types.doc.labels import DocItemLabel

        self._log(f"\nProcessing {pdf_path.name}")
        try:
            result = self.converter.convert(str(pdf_path))
            document = result.document
        except Exception as exc:
            self._log(f"Conversion failed: {exc}")
            return []

        try:
            chunks = list(self._create_chunker().chunk(dl_doc=document))
        except Exception as exc:
            self._log(f"Chunking failed: {exc}")
            return []

        self._log(f"Generated {len(chunks)} chunks")
        processed_chunks: list[ProcessedChunk] = []

        for index, chunk in enumerate(chunks):
            headings: list[str] = []
            page_number: Optional[int] = None
            chunk_type = "text"
            metadata = getattr(chunk, "meta", None)

            if metadata:
                raw_headings = getattr(metadata, "headings", None) or []
                headings = [heading for heading in raw_headings if heading]

                for item_ref in getattr(metadata, "doc_items", None) or []:
                    try:
                        for provenance in getattr(item_ref, "prov", None) or []:
                            if hasattr(provenance, "page_no"):
                                page_number = provenance.page_no
                                break
                    except Exception:
                        pass

                    try:
                        label = getattr(item_ref, "label", None)
                        if label == DocItemLabel.TABLE:
                            chunk_type = "table"
                        elif label == DocItemLabel.LIST_ITEM:
                            chunk_type = "list"
                        elif label == DocItemLabel.PICTURE:
                            chunk_type = "figure"
                        elif label == DocItemLabel.FORMULA:
                            chunk_type = "formula"
                    except Exception:
                        pass

            chunk_text = getattr(chunk, "text", str(chunk))
            if not chunk_text or not chunk_text.strip():
                continue

            processed_chunks.append(
                ProcessedChunk(
                    text=chunk_text,
                    source_file=pdf_path.name,
                    page=page_number,
                    chunk_index=index,
                    headings=headings,
                    chunk_type=chunk_type,
                    token_count=int(len(chunk_text.split()) * 1.3),
                )
            )

        content_types: dict[str, int] = {}
        for chunk in processed_chunks:
            content_types[chunk.chunk_type] = (
                content_types.get(chunk.chunk_type, 0) + 1
            )
        self._log(f"Content types: {content_types}")
        return processed_chunks


class DoclingIndexer:
    """Create and update a Milvus index from PDF reports."""

    def __init__(
        self,
        pdf_folder: str | Path,
        db_path: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_tokens: int = 512,
        device: str = "cpu",
        verbose: bool = True,
        assume_yes: bool = False,
    ) -> None:
        self.pdf_folder = Path(pdf_folder).expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.device = device
        self.verbose = verbose
        self.assume_yes = assume_yes

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _prompt_continue_after_pdf(
        self,
        basename: str,
        chunks_indexed: int,
        pdfs_remaining: int,
        *,
        full_initial_build: bool,
    ) -> bool:
        """Prompt before processing the next PDF in an interactive run."""

        if (
            pdfs_remaining <= 0
            or chunks_indexed <= 0
            or self.assume_yes
            or not sys.stdin.isatty()
        ):
            return True

        try:
            answer = input(
                f"Continue with the next PDF? "
                f"(finished {basename}: {chunks_indexed} chunks) [Y/n]: "
            ).strip().lower()
        except EOFError:
            self._log("End of input; stopping.")
            return False

        if answer not in {"n", "no", "q", "quit"}:
            return True

        if full_initial_build:
            self._log(
                "The database is partly filled. Re-run with --incremental "
                "to add the remaining PDFs."
            )
        else:
            self._log(
                "Re-run with --incremental to continue; indexed PDFs are skipped."
            )
        return False

    def _index_pdf(
        self,
        processor: DoclingProcessor,
        rag_index: Any,
        pdf_path: Path,
    ) -> tuple[int, int]:
        """Parse and index one PDF, returning chunk and table counts."""

        from tqdm import tqdm
        from verbatim_rag.document import DocumentType

        chunks = processor.process_pdf(pdf_path)
        if not chunks:
            self._log(f"No chunks generated for {pdf_path.name}")
            return 0, 0

        chunks_added = 0
        table_chunks = 0
        for chunk in tqdm(
            chunks,
            desc=f"Indexing {pdf_path.name}",
            leave=False,
        ):
            if chunk.chunk_type == "table":
                table_chunks += 1

            doc_id = f"{chunk.source_file}_chunk_{chunk.chunk_index}"
            try:
                rag_index.add_document(
                    content=chunk.text,
                    metadata={
                        "source_file": chunk.source_file,
                        "page": chunk.page if chunk.page is not None else 0,
                        "chunk_index": chunk.chunk_index,
                        "headings": (
                            "|".join(chunk.headings) if chunk.headings else ""
                        ),
                        "chunk_type": chunk.chunk_type,
                        "token_count": chunk.token_count,
                    },
                    doc_id=doc_id,
                    document_type=DocumentType.PDF,
                )
                chunks_added += 1
            except Exception as exc:
                self._log(
                    f"Failed to index chunk {chunk.chunk_index} "
                    f"from {pdf_path.name}: {exc}"
                )

        return chunks_added, table_chunks

    def _build_index_components(
        self,
        *,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        dense_dim: Optional[int] = None,
    ) -> tuple[Any, Any]:
        """Create embedding providers, vector store, and Verbatim index."""

        from verbatim_rag.embedding_providers import SpladeProvider
        from verbatim_rag.index import VerbatimIndex
        from verbatim_rag.vector_stores import LocalMilvusStore

        dense_embedder = (
            create_embedder(self.embedding_model, self.device)
            if enable_dense
            else None
        )
        if enable_dense:
            model_dimension = dense_embedder.get_dimension()
            if dense_dim is not None and model_dimension != dense_dim:
                raise ValueError(
                    "Dense dimension mismatch: "
                    f"model outputs {model_dimension}, index expects {dense_dim}."
                )
            dense_dim = model_dimension

        sparse_provider = (
            SpladeProvider(model_name=DEFAULT_SPARSE_MODEL, device=self.device)
            if enable_sparse
            else None
        )

        store = LocalMilvusStore(
            str(self.db_path),
            enable_sparse=enable_sparse,
            enable_dense=enable_dense,
            dense_dim=dense_dim or 384,
        )
        rag_index = VerbatimIndex(
            vector_store=store,
            dense_provider=dense_embedder,
            sparse_provider=sparse_provider,
        )
        return store, rag_index

    def _pdf_files(self) -> list[Path]:
        """Return all PDFs in the configured source directory."""

        if not self.pdf_folder.is_dir():
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_folder}")

        files = sorted(self.pdf_folder.glob("*.pdf"))
        if not files:
            raise FileNotFoundError(
                f"No PDF files found in {self.pdf_folder}"
            )
        return files

    def create_index(self, force: bool = False) -> bool:
        """Create a new Milvus index, replacing an existing database."""

        from pymilvus import connections

        print("=" * 70)
        print("DOCLING MILVUS INDEX CREATION")
        print("=" * 70)
        print(f"PDF source: {self.pdf_folder}")
        print(f"Database: {self.db_path}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Maximum tokens per chunk: {self.max_tokens}")

        if not force and not confirm_deletion(self.db_path, self.assume_yes):
            print("Operation cancelled.")
            return False

        if self.db_path.is_file():
            self.db_path.unlink()
        elif self.db_path.is_dir():
            shutil.rmtree(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            connections.disconnect("default")
        except Exception:
            pass

        processor = DoclingProcessor(
            embedding_model_id=self.embedding_model,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        _, rag_index = self._build_index_components()
        pdf_files = self._pdf_files()

        total_chunks = 0
        total_tables = 0
        for index, pdf_path in enumerate(pdf_files):
            chunks_added, table_chunks = self._index_pdf(
                processor,
                rag_index,
                pdf_path,
            )
            total_chunks += chunks_added
            total_tables += table_chunks
            self._log(
                f"Finished {pdf_path.name}: {chunks_added} chunks "
                f"({table_chunks} table chunks)."
            )
            if not self._prompt_continue_after_pdf(
                pdf_path.name,
                chunks_added,
                len(pdf_files) - index - 1,
                full_initial_build=True,
            ):
                break

        print("\nIndexing complete")
        print(f"Total chunks indexed: {total_chunks}")
        print(f"Table chunks indexed: {total_tables}")
        print(f"Database: {self.db_path}")
        return True

    def append_index(
        self,
        reindex_all: bool = False,
        reindex_basenames: Optional[set[str]] = None,
        skip_existing: bool = True,
    ) -> bool:
        """Append new PDFs or replace selected PDFs in an existing index."""

        from pymilvus import MilvusClient, connections

        print("=" * 70)
        print("INCREMENTAL MILVUS INDEX UPDATE")
        print("=" * 70)
        print(f"PDF source: {self.pdf_folder}")
        print(f"Database: {self.db_path}")

        pdf_files = self._pdf_files()
        folder_names = {path.name for path in pdf_files}
        if reindex_basenames:
            unknown = reindex_basenames - folder_names
            if unknown:
                self._log(
                    "Requested PDFs not found and will be skipped: "
                    + ", ".join(sorted(unknown))
                )

        probe = MilvusClient(str(self.db_path))
        collection_exists = False
        dense_dim = 384
        enable_dense = True
        enable_sparse = True
        try:
            collection_exists = probe.has_collection(DEFAULT_COLLECTION_NAME)
            if collection_exists:
                dense_dim, enable_dense, enable_sparse = (
                    infer_vector_store_settings(
                        probe,
                        DEFAULT_COLLECTION_NAME,
                    )
                )
        finally:
            probe.close()

        if not collection_exists:
            self._log("No existing collection found; starting a full build.")
            return self.create_index(force=False)

        try:
            connections.disconnect("default")
        except Exception:
            pass

        processor = DoclingProcessor(
            embedding_model_id=self.embedding_model,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        store, rag_index = self._build_index_components(
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            dense_dim=dense_dim,
        )

        total_chunks = 0
        total_tables = 0
        pdfs_updated = 0
        for index, pdf_path in enumerate(pdf_files):
            name = pdf_path.name
            if reindex_all or (
                reindex_basenames and name in reindex_basenames
            ):
                removed = delete_chunks_for_pdf(store, name)
                self._log(f"Removed {removed} existing chunks for {name}.")
            elif skip_existing and pdf_has_indexed_chunks(store, name):
                self._log(f"Skipping already indexed PDF: {name}")
                continue

            chunks_added, table_chunks = self._index_pdf(
                processor,
                rag_index,
                pdf_path,
            )
            total_chunks += chunks_added
            total_tables += table_chunks
            if chunks_added:
                pdfs_updated += 1

            self._log(
                f"Finished {name}: {chunks_added} chunks "
                f"({table_chunks} table chunks)."
            )
            if not self._prompt_continue_after_pdf(
                name,
                chunks_added,
                len(pdf_files) - index - 1,
                full_initial_build=False,
            ):
                break

        print("\nIncremental update complete")
        print(f"New chunks indexed: {total_chunks}")
        print(f"Table chunks indexed: {total_tables}")
        print(f"PDF files updated: {pdfs_updated}")
        print(f"Database: {self.db_path}")
        return True


def create_index(
    pdf_folder: str | Path,
    db_path: str | Path,
    device: str = "cpu",
    force: bool = False,
    max_tokens: int = 512,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    assume_yes: bool = False,
) -> bool:
    """Create a new index, replacing an existing database."""

    return DoclingIndexer(
        pdf_folder=pdf_folder,
        db_path=db_path,
        embedding_model=embedding_model,
        max_tokens=max_tokens,
        device=device,
        verbose=True,
        assume_yes=assume_yes,
    ).create_index(force=force)


def append_to_index(
    pdf_folder: str | Path,
    db_path: str | Path,
    device: str = "cpu",
    max_tokens: int = 512,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reindex_all: bool = False,
    reindex_basenames: Optional[set[str]] = None,
    skip_existing: bool = True,
    assume_yes: bool = False,
) -> bool:
    """Add or replace PDFs in an existing index."""

    return DoclingIndexer(
        pdf_folder=pdf_folder,
        db_path=db_path,
        embedding_model=embedding_model,
        max_tokens=max_tokens,
        device=device,
        verbose=True,
        assume_yes=assume_yes,
    ).append_index(
        reindex_all=reindex_all,
        reindex_basenames=reindex_basenames,
        skip_existing=skip_existing,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Create a Milvus index from PDFs using Docling."
    )
    parser.add_argument(
        "--pdf-folder",
        default=str(DEFAULT_PDF_FOLDER),
        help=(
            "Directory containing PDF reports "
            f"(default: {DEFAULT_PDF_FOLDER})"
        ),
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Milvus Lite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Replace an existing database without confirmation.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Add PDFs to an existing database and skip indexed files.",
    )
    parser.add_argument(
        "--reindex",
        nargs="*",
        default=None,
        metavar="PDF",
        help=(
            "With --incremental, replace the listed PDF basenames. "
            "Pass --reindex without names to replace every PDF."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run non-interactively and accept destructive confirmation.",
    )
    return parser


def main() -> None:
    """Run the index creation CLI."""

    args = build_parser().parse_args()

    if args.incremental and args.reindex is not None:
        reindex_all = len(args.reindex) == 0
        reindex_basenames = (
            None if reindex_all else set(args.reindex)
        )
    else:
        reindex_all = False
        reindex_basenames = None

    if args.incremental:
        success = append_to_index(
            pdf_folder=args.pdf_folder,
            db_path=args.db_path,
            device=args.device,
            max_tokens=args.max_tokens,
            embedding_model=args.embedding_model,
            reindex_all=reindex_all,
            reindex_basenames=reindex_basenames,
            skip_existing=True,
            assume_yes=args.yes,
        )
    else:
        success = create_index(
            pdf_folder=args.pdf_folder,
            db_path=args.db_path,
            device=args.device,
            force=args.force,
            max_tokens=args.max_tokens,
            embedding_model=args.embedding_model,
            assume_yes=args.yes,
        )

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
