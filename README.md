# Internal Knowledge Base Chatbot (RAG)

A backend API that answers questions grounded in your company's internal documents,
using Retrieval-Augmented Generation.

## How it works

```
                 ┌───────────────┐
 documents/  ───▶│   Ingestion   │──▶ chunks ──▶ embeddings ──▶ Chroma vector store
 (.pdf/.docx/     │ (app/ingestion)                                     │
  .txt/.md)       └───────────────┘                                    │
                                                                        ▼
 User question ──▶ /chat endpoint ──▶ retriever (top-k similar chunks) │
                                              │                        │
                                              ▼                        │
                                     grounded prompt + Claude ◀────────┘
                                              │
                                              ▼
                                   answer + cited sources
```

- **Embeddings**: local `sentence-transformers` model — free, runs on CPU, no API key needed just to index documents.
- **Vector store**: [Chroma](https://www.trychroma.com/), persisted to disk at `data/vectorstore`.
- **Generation**: Google Gemini (via `langchain-google-genai`), prompted to answer *only* from retrieved context and cite sources. Gemini has a genuinely free tier (rate-limited, no credit card needed).
- **Memory**: simple in-memory per-session conversation history (swap for Redis/DB for multi-instance production use).

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=AIza... (free key from https://aistudio.google.com/apikey)
```

## Add your documents

Drop `.pdf`, `.docx`, `.txt`, or `.md` files into `data/documents/` (a sample HR policy file
is included so you can test the pipeline immediately).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Then build the index (do this once, and again any time documents change):

```bash
curl -X POST http://localhost:8000/ingest
```

Or run ingestion directly from the command line without starting the server:

```bash
python -m app.ingestion
```

## Usage

**Ask a question:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many remote days per week are employees allowed?"}'
```

Response:
```json
{
  "answer": "Employees may work remotely up to 3 days per week, subject to manager approval [source: sample_hr_policy.txt].",
  "sources": ["sample_hr_policy.txt"],
  "session_id": "b3f1c2..."
}
```

**Continue the conversation** (pass back the `session_id` to keep memory):
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What about the home office stipend?", "session_id": "b3f1c2..."}'
```

**Clear a session:**
```bash
curl -X DELETE http://localhost:8000/chat/b3f1c2...
```

**Interactive API docs** (Swagger UI): visit `http://localhost:8000/docs` once the server is running.

## Securing the API (optional)

Set `APP_API_KEY` in `.env` to require an `X-API-Key` header on every request. Leave it blank
to disable auth (useful for local dev only — never leave it blank in production).

## Configuration reference

All settings live in `app/config.py` and can be overridden via environment variables — see `.env.example`.

| Variable | Purpose | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio (Gemini) API key (required) | — |
| `LLM_MODEL` | Gemini model for generation | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Sentence-transformers model for retrieval | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text splitting parameters | `1000` / `150` |
| `RETRIEVAL_TOP_K` | Number of chunks retrieved per question | `4` |
| `MAX_HISTORY_TURNS` | Conversation turns kept per session | `6` |
| `APP_API_KEY` | Optional shared-secret auth for the API | disabled |

## Extending this

- **Swap the vector store**: replace Chroma with Pinecone/Weaviate/Qdrant by changing `app/ingestion.py` — the rest of the app is unaffected since it only talks to a LangChain retriever interface.
- **Swap embeddings**: switch to `OpenAIEmbeddings` or Voyage AI embeddings for higher retrieval quality at the cost of an API dependency.
- **Add re-ranking**: insert a cross-encoder reranker step between retrieval and generation for better precision on large corpora.
- **Add auth/rate limiting**: put this behind an API gateway (or add FastAPI middleware) before exposing it beyond internal use.
- **Persistent sessions**: replace the in-memory `_sessions` dict in `app/main.py` with Redis for multi-instance deployments.

## Notes on this sandbox build

This code was written and syntax-checked in an offline sandbox (no network access to
verify pip installs or make live Gemini API calls). Before relying on it, run through
the Setup steps above in an environment with internet access, and test end-to-end with
your own API key and documents.
