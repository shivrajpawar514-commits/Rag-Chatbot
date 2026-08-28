"""
Central configuration for the RAG chatbot.
All values can be overridden via environment variables (see .env.example).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Paths ---
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "data" / "vectorstore"))

# --- LLM (generation) ---
# Get a free key at https://aistudio.google.com/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Embeddings (retrieval) ---
# Local, free, no API key required. Swap for OpenAIEmbeddings if you prefer.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_knowledge_base")

# --- API ---
API_KEY = os.getenv("APP_API_KEY", "")  # optional shared secret to protect the API itself
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# --- Dynamic Configuration for Student Project Demo ---
import json

CONFIG_FILE = BASE_DIR / "data" / "project_config.json"

DEFAULT_SYSTEM_PROMPT = """You are an internal knowledge assistant for the company. Answer the user's question using ONLY the information in the CONTEXT below.

Rules:
- If the context does not contain enough information to answer, say so plainly instead of guessing or using outside knowledge.
- Be concise and direct. Use bullet points for lists.
- When you use a fact from a source, cite it inline like [source: <filename>].
- Never fabricate a source or a fact that isn't in the context.

CONTEXT:
{context}
"""

_config_state = {
    "llm_model": LLM_MODEL,
    "llm_temperature": LLM_TEMPERATURE,
    "llm_max_tokens": LLM_MAX_TOKENS,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "system_prompt": DEFAULT_SYSTEM_PROMPT.strip(),
    "rag_mode": "standard",
    "rerank_enabled": False,
    "query_router_enabled": True,
    "student_name": "Jane Doe",
    "student_roll": "CS-2026-089",
    "student_course": "B.Tech Computer Science & Engineering",
    "project_guide": "Dr. Alan Turing"
}

def load_dynamic_config():
    global LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    _config_state[k] = v
        except Exception:
            pass
    
    # Expose state as globals for compatibility
    LLM_MODEL = _config_state["llm_model"]
    LLM_TEMPERATURE = float(_config_state["llm_temperature"])
    LLM_MAX_TOKENS = int(_config_state["llm_max_tokens"])
    CHUNK_SIZE = int(_config_state["chunk_size"])
    CHUNK_OVERLAP = int(_config_state["chunk_overlap"])
    RETRIEVAL_TOP_K = int(_config_state["retrieval_top_k"])

def save_dynamic_config(new_settings: dict):
    for k, v in new_settings.items():
        if k in _config_state:
            if k == "llm_temperature":
                _config_state[k] = float(v)
            elif k in ["llm_max_tokens", "chunk_size", "chunk_overlap", "retrieval_top_k"]:
                _config_state[k] = int(v)
            elif k in ["rerank_enabled", "query_router_enabled"]:
                # Parse boolean safely from string or boolean type
                if isinstance(v, str):
                    _config_state[k] = v.lower() == "true"
                else:
                    _config_state[k] = bool(v)
            else:
                _config_state[k] = str(v)
    
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(_config_state, f, indent=2)
    load_dynamic_config()

# Load settings at import time
load_dynamic_config()

