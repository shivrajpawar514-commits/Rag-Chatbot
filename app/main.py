"""
FastAPI backend for the Academic RAG chatbot.
Includes diagnostic tools, parameters configurations, chunk explorer, and file manager.

Run:
    uvicorn app.main:app --reload --port 8000
"""
import logging
import os
import shutil
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Header, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.ingestion import ingest as run_ingestion
from app.rag_chain import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GroundDoc-RAG: Academic Project Dashboard",
    description="RAG-powered system with dynamic chunking, evaluations, and interactive diagnostics.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store: session_id -> deque of (role, content).
_sessions: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=config.MAX_HISTORY_TURNS * 2)
)


# ---------- Schemas ----------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    session_id: Optional[str] = Field(
        None, description="Reuse to maintain conversation memory. Omit to start fresh."
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    diagnostics: Optional[dict] = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


# ---------- Auth (optional shared-secret) ----------

def _check_api_key(x_api_key: Optional[str]):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(x_api_key: Optional[str] = Header(None)):
    """
    Re-scans data/documents, re-chunks, re-embeds, and rebuilds the vector store.
    """
    _check_api_key(x_api_key)
    try:
        n_chunks = run_ingestion()
        # Force reload vector store in pipeline
        get_pipeline().refresh_vectorstore()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if n_chunks == 0:
        return IngestResponse(chunks_indexed=0, message="No supported documents found to index.")
    return IngestResponse(chunks_indexed=n_chunks, message="Vector store rebuilt successfully.")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_api_key: Optional[str] = Header(None)):
    """
    Ask a question. Returns answer, sources, and detailed diagnostics.
    """
    _check_api_key(x_api_key)

    session_id = request.session_id or str(uuid.uuid4())
    history = list(_sessions[session_id])

    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = pipeline.ask(request.question, history=history)
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    _sessions[session_id].append(("human", request.question))
    _sessions[session_id].append(("ai", result["answer"]))

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
        diagnostics=result.get("diagnostics")
    )


@app.delete("/chat/{session_id}")
def clear_session(session_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    return {"cleared": existed}


# ---------- Academic Project Specific APIs ----------

@app.get("/api/documents")
def list_documents(x_api_key: Optional[str] = Header(None)):
    """Lists files in data/documents/ with details and their chunk counts in Chroma."""
    _check_api_key(x_api_key)
    doc_dir = config.DOCUMENTS_DIR
    if not doc_dir.exists():
        doc_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    chunk_counts = {}
    try:
        pipeline = get_pipeline()
        db = pipeline.vectorstore
        db_data = db.get()
        if db_data and "metadatas" in db_data and db_data["metadatas"]:
            for meta in db_data["metadatas"]:
                source = meta.get("source")
                if source:
                    chunk_counts[source] = chunk_counts.get(source, 0) + 1
    except Exception as e:
        logger.warning("Could not load chunk counts from database: %s", e)

    for entry in os.scandir(doc_dir):
        if entry.is_file() and not entry.name.startswith('.'):
            stat = entry.stat()
            suffix = os.path.splitext(entry.name)[1].lower()
            files.append({
                "name": entry.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "chunks_count": chunk_counts.get(entry.name, 0),
                "extension": suffix
            })
    return {"documents": sorted(files, key=lambda x: x["name"])}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """Uploads a PDF, DOCX, TXT, or MD file to the documents folder."""
    _check_api_key(x_api_key)
    allowed_exts = {".pdf", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{ext}'. Supported formats: PDF, DOCX, TXT, MD"
        )
    
    doc_dir = config.DOCUMENTS_DIR
    doc_dir.mkdir(parents=True, exist_ok=True)
    target_path = doc_dir / file.filename
    
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {e}")
        
    return {"filename": file.filename, "message": "Document uploaded successfully."}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str, x_api_key: Optional[str] = Header(None)):
    """Deletes a file from the documents folder and clears its chunks from the database."""
    _check_api_key(x_api_key)
    target_path = config.DOCUMENTS_DIR / filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    
    try:
        os.remove(target_path)
    except Exception as e:
        logger.exception("Failed to delete file")
        raise HTTPException(status_code=500, detail=f"Failed to delete file from disk: {e}")
        
    # Clear matching chunks from Chroma
    try:
        pipeline = get_pipeline()
        db = pipeline.vectorstore
        # We can delete chunks that have the source matching the filename
        db.delete(where={"source": filename})
    except Exception as e:
        logger.warning("Could not delete vector db chunks for document %s: %s", filename, e)
        
    return {"filename": filename, "message": "Document deleted and index updated."}


@app.get("/api/config")
def get_sys_config(x_api_key: Optional[str] = Header(None)):
    """Returns the current runtime settings."""
    _check_api_key(x_api_key)
    return config._config_state


@app.post("/api/config")
def update_sys_config(settings: dict, x_api_key: Optional[str] = Header(None)):
    """Updates settings dynamically."""
    _check_api_key(x_api_key)
    try:
        config.save_dynamic_config(settings)
        return {"status": "success", "message": "Parameters updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update parameters: {e}")


@app.get("/api/chunks")
def get_db_chunks(query: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    """Fetches all indexed chunks from the database for the Chunk Explorer."""
    _check_api_key(x_api_key)
    try:
        pipeline = get_pipeline()
        db = pipeline.vectorstore
        db_data = db.get()
        chunks = []
        if db_data and "documents" in db_data and db_data["documents"]:
            for idx, text in enumerate(db_data["documents"]):
                source = db_data["metadatas"][idx].get("source", "unknown") if db_data["metadatas"] else "unknown"
                chunk_id = db_data["ids"][idx] if db_data["ids"] else str(idx)
                
                # Apply filter query if any
                if query and query.lower() not in text.lower() and query.lower() not in source.lower():
                    continue
                    
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "source": source
                })
        return {"chunks": chunks}
    except Exception as e:
        logger.exception("Failed to get DB chunks")
        return {"chunks": [], "error": str(e)}


# Serve the chat UI at the root URL. Mounted last so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
