# GroundDoc-RAG: Advanced Retrieval-Augmented Generation System & Dashboard

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** platform and **Academic Project Dashboard** for question-answering over unstructured documents. Built with **FastAPI**, **LangChain**, **Google Gemini**, **ChromaDB**, and **Sentence-Transformers**.

---

## 🌟 Key Highlights & Advanced Features

- **Cognitive Query Routing**: Uses LLM classification to automatically route queries to `GENERAL` (direct response, 0ms retrieval latency), `SUMMARY` (multi-document synthesis), or `RAG` (targeted vector/keyword search).
- **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Merges dense semantic embeddings (`all-MiniLM-L6-v2`) with sparse lexical matching (zero-dependency BM25) using standard RRF formulas:
  $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{Rank}_m(d)}$$
- **Multi-Query Expansion**: Dynamically expands queries into 3 alternative formulations to maximize semantic coverage and recall across document indexes.
- **Keyword-Reweighted Reranking**: Boosts chunk relevance scores dynamically based on exact query term frequency and overlap.
- **Interactive Academic Dashboard (`/`)**:
  - **Chat Sandbox & RAG Debugger**: Live chat interface alongside a real-time diagnostics console displaying query latency splits (retrieval vs. synthesis), exact prompt payloads, retrieved chunk weights, and **RAG Triad** evaluation scores (Context Relevance, Groundedness, Answer Relevance).
  - **Document Hub**: Drag-and-drop file uploader (PDF, DOCX, TXT, MD) with multi-stage index synchronization.
  - **Vector Chunk Inspector**: Direct queryable viewer to inspect text chunk partitions stored inside ChromaDB.
  - **Engine Hyperparameters**: Live sliders and controls for Temperature, Top-K, Chunk Size, Overlap, Retrieval Mode, and System Prompt templates.
  - **Project Metadata & Architecture**: Editable academic credentials (Student Name, Roll No, Supervisor) with interactive Mermaid.js system architecture flowcharts.

---

## 🏗️ System Architecture

```
                                  [ User Document Uploads ]
                                  (PDF, DOCX, TXT, MD files)
                                              │
                                              ▼
                                 [ Recursive Text Splitter ]
                                   (Chunk Size & Overlap)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [ Dense Embedding Model ]                             [ Sparse BM25 Index ]
      (all-MiniLM-L6-v2 / 384-dim)                            (Term Frequency & IDF)
                    │                                                   │
                    ▼                                                   │
         [( ChromaDB Vector Store )]                                    │
                    │                                                   │
════════════════════╪═══════════════════════════════════════════════════╪══════════════════════════
                    │                ONLINE QUERY PIPELINE              │
                    │                                                   │
                    │                [ User Question ]                  │
                    │                        │                          │
                    │                        ▼                          │
                    │              { Query Router / LLM }               │
                    │             /          │         \                │
           GENERAL (Bypass)      /     SUMMARY      \   RAG (Standard)  │
                  │             /    (Fetch Full)    \                  │
                  │            /          │           \                 │
                  │           │           │     [ Multi-Query Expansion ]
                  │           │           │            │
                  │           │           │            ├───────────────┐
                  │           ▼           ▼            ▼               ▼
                  │     [( Dense Vector Match )]  [( Dense Match )]  [ BM25 Search ]
                  │               │                    │               │
                  │               │                    └───────┬───────┘
                  │               │                            ▼
                  │               │                 [ Reciprocal Rank Fusion ]
                  │               │                            │
                  │               │                            ▼
                  │               │                 [ Keyword Reranker ]
                  │               │                            │
                  │               └───────────┬────────────────┘
                  │                           ▼
                  │               [ Grounded Prompt Formatter ]
                  │                           │
                  └───────────────────────────┼────────────────────────┘
                                              ▼
                                   [ Google Gemini LLM ]
                                              │
                                              ▼
                               [ Grounded Response + Sources ]
```

---

## 📁 Repository Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py             # Dynamic configuration & persistent settings manager
│   ├── ingestion.py          # Document loaders, text splitters, and ChromaDB builder
│   ├── main.py               # FastAPI backend with REST API & static file serving
│   ├── rag_chain.py          # Advanced RAG pipeline (BM25, RRF, Router, Expansion, Reranker)
│   └── static/
│       └── index.html        # Modern dark-mode web dashboard & telemetry UI
├── data/
│   ├── documents/            # Source documents (PDF, DOCX, TXT, MD)
│   ├── project_config.json   # Persisted runtime hyperparameters and author metadata
│   └── vectorstore/          # Chroma vector database storage directory
├── requirements.txt          # Python dependencies
├── .env.example              # Sample environment configuration
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone repository
git clone https://github.com/shivrajpawar514-commits/Rag-Chatbot.git
cd Rag-Chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Obtain a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
# Create .env file from template
cp .env.example .env
```

Edit `.env` and set your key:
```ini
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Application

```bash
uvicorn app.main:app --reload --port 8000
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status check |
| `POST` | `/chat` | Submit a question with optional session history memory and receive grounded answers + diagnostics |
| `DELETE`| `/chat/{session_id}` | Clear conversation memory for a specific session |
| `POST` | `/ingest` | Re-scans `data/documents/`, splits text, generates embeddings, and rebuilds ChromaDB index |
| `GET` | `/api/documents` | List all files in the catalog along with disk size and indexed chunk counts |
| `POST` | `/api/documents/upload` | Upload new PDF, DOCX, TXT, or MD documents |
| `DELETE`| `/api/documents/{filename}` | Delete a document from disk and purge associated vector records |
| `GET` | `/api/config` | Retrieve current runtime hyperparameter settings |
| `POST` | `/api/config` | Update and persist hyperparameters dynamically |
| `GET` | `/api/chunks` | Search and explore raw vector chunks stored in ChromaDB |

---

## 📊 RAG Triad Evaluation Framework

The system computes heuristic quality metrics in real-time for every response:

1. **Context Relevance**: Evaluates whether retrieved document chunks contain the requisite information to answer the query without extraneous noise.
2. **Groundedness (Faithfulness)**: Measures if claims in the LLM's answer are strictly grounded in the retrieved context documents to prevent hallucinations.
3. **Answer Relevance**: Checks whether the generated response directly addresses the user's input prompt.

---

## ⚙️ Configuration Reference

All settings can be configured dynamically through the UI or defined in `.env`:

| Parameter | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini API Key | Required |
| `LLM_MODEL` | Generator model (`gemini-2.5-flash`, `gemini-2.5-pro`) | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Local Sentence-Transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Recursive character text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks | `150` |
| `RETRIEVAL_TOP_K` | Number of context chunks fed to the LLM | `4` |
| `rag_mode` | Retrieval mode (`standard`, `hybrid`, `cognitive`) | `standard` |
| `query_router_enabled` | Enable/disable LLM query routing classifier | `true` |
| `rerank_enabled` | Enable/disable keyword-reweighted reranker | `false` |

---


