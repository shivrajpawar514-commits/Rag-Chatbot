"""
Ingestion pipeline: load raw documents -> split into chunks -> embed -> persist to Chroma.

Run directly to (re)build the vector store from everything in DOCUMENTS_DIR:
    python -m app.ingestion
"""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_documents(source_dir: Path) -> List[Document]:
    """Load every supported file under source_dir into LangChain Document objects."""
    docs: List[Document] = []
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {source_dir}")

    files = [f for f in source_dir.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_MAP]

    if not files:
        logger.warning("No supported files (%s) found in %s", list(LOADER_MAP.keys()), source_dir)
        return docs

    for path in files:
        loader_cls = LOADER_MAP[path.suffix.lower()]
        try:
            loaded = loader_cls(str(path)).load()
            for d in loaded:
                d.metadata["source"] = path.name
            docs.extend(loaded)
            logger.info("Loaded %s (%d page(s)/section(s))", path.name, len(loaded))
        except Exception as e:
            logger.error("Failed to load %s: %s", path.name, e)

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks sized for retrieval."""
    chunk_size = config._config_state.get("chunk_size", config.CHUNK_SIZE)
    chunk_overlap = config._config_state.get("chunk_overlap", config.CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d document(s) into %d chunk(s) using size=%d, overlap=%d", len(docs), len(chunks), chunk_size, chunk_overlap)
    return chunks


_embedding_instance = None

def get_embedding_function() -> HuggingFaceEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        logger.info("Loading HuggingFaceEmbeddings model: %s...", config.EMBEDDING_MODEL)
        _embedding_instance = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embedding_instance



def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them to the Chroma vector store on disk."""
    embeddings = get_embedding_function()
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    logger.info("Persisted vector store to %s", config.VECTORSTORE_DIR)
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an already-built vector store from disk (does not re-embed)."""
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )


def ingest(source_dir: Path = None) -> int:
    """Full pipeline: load -> chunk -> embed -> persist. Returns number of chunks indexed."""
    source_dir = Path(source_dir) if source_dir else config.DOCUMENTS_DIR
    docs = load_documents(source_dir)
    if not docs:
        return 0
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingestion complete. {n} chunks indexed into '{config.COLLECTION_NAME}'.")
