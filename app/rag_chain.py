"""
Builds an advanced retrieval-augmented generation chain with:
- Query Routing (GENERAL vs. SUMMARY vs. RAG)
- Multi-Query Expansion
- Hybrid Search (Dense Vector + Sparse BM25 Keyword Search)
- Reciprocal Rank Fusion (RRF)
- Heuristic Keyword-Reweighted Reranker
- Full execution logging and diagnostic telemetry
"""
import time
import re
import math
from typing import List, Tuple, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app import config
from app.ingestion import load_vectorstore

# ---------- Zero-Dependency BM25 Sparse Keyword Search ----------

class SimpleBM25:
    """
    A pure Python implementation of the BM25 retrieval algorithm.
    Allows hybrid sparse-dense retrieval without external search engine overhead.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = []
        self.doc_term_freqs = []
        
        # Count document frequencies (DF) for IDF computation
        df = {}
        total_len = 0
        
        for item in corpus:
            text = item["text"].lower()
            # Simple tokenization: keep words of alphanumeric characters
            tokens = re.findall(r'\b\w+\b', text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            self.doc_term_freqs.append(term_freq)
            
            # Increment document frequencies
            for token in term_freq.keys():
                df[token] = df.get(token, 0) + 1
                
        self.avg_doc_len = total_len / max(1, self.corpus_size)
        
        # Calculate IDF for all seen terms
        self.idf = {}
        for term, count in df.items():
            # Standard BM25 IDF formulation
            self.idf[term] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        """Compute BM25 scores for the query against all documents."""
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        scores = []
        
        for i in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lens[i]
            term_freqs = self.doc_term_freqs[i]
            
            for token in query_tokens:
                if token in self.idf:
                    tf = term_freqs.get(token, 0)
                    idf = self.idf[token]
                    
                    # BM25 tf scaling
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            scores.append(score)
            
        return scores


# ---------- Helper Utilities ----------

def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no relevant documents found)"
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        blocks.append(f"--- from {source} ---\n{d.page_content}")
    return "\n\n".join(blocks)


def calculate_rag_metrics(question: str, context: str, answer: str) -> Dict[str, float]:
    """
    Calculate deterministic heuristic RAG evaluation metrics (0 to 100).
    """
    def get_keywords(text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return set(words)

    q_words = get_keywords(question)
    c_words = get_keywords(context)
    a_words = get_keywords(answer)

    # 1. Context Relevance
    if not q_words:
        context_relevance = 100.0
    else:
        overlap = q_words.intersection(c_words)
        ratio = len(overlap) / len(q_words) if len(q_words) > 0 else 1.0
        context_relevance = 65.0 + (ratio * 35.0)

    # 2. Groundedness
    if not a_words:
        groundedness = 100.0
    else:
        overlap = a_words.intersection(c_words)
        overlap = overlap.union(q_words.intersection(a_words))
        ratio = len(overlap) / len(a_words) if len(a_words) > 0 else 1.0
        if "not mention" in answer.lower() or "cannot find" in answer.lower() or "not in the context" in answer.lower():
            groundedness = 98.0
        else:
            groundedness = 70.0 + (ratio * 30.0)

    # 3. Answer Relevance
    if not q_words or not a_words:
        answer_relevance = 100.0
    else:
        overlap = q_words.intersection(a_words)
        ratio = len(overlap) / len(q_words) if len(q_words) > 0 else 1.0
        if "not mention" in answer.lower() or "cannot find" in answer.lower() or "not in the context" in answer.lower():
            answer_relevance = 45.0
        else:
            answer_relevance = 75.0 + (ratio * 25.0)

    return {
        "context_relevance": round(min(100.0, max(10.0, context_relevance)), 1),
        "groundedness": round(min(100.0, max(10.0, groundedness)), 1),
        "answer_relevance": round(min(100.0, max(10.0, answer_relevance)), 1),
    }


# ---------- Advanced RAG Pipeline ----------

class RAGPipeline:
    """Wraps retriever + LLM chain with query routing, hybrid search, RRF and reranker."""

    def __init__(self):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your environment or .env file."
            )
        self._vectorstore = None
        self._bm25_instance = None
        self._corpus_cache = [] # Cache items: {"text": str, "source": str, "doc_obj": Document, "id": str}

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = load_vectorstore()
        return self._vectorstore

    def refresh_vectorstore(self):
        """Force reload the vector store and clear BM25 cache."""
        self._vectorstore = None
        self._bm25_instance = None
        self._corpus_cache = []

    def _build_bm25_index(self):
        """Load all chunks from the active vector store and build the BM25 index."""
        if self._bm25_instance is not None:
            return
        
        try:
            db_data = self.vectorstore.get()
            corpus = []
            self._corpus_cache = []
            
            if db_data and "documents" in db_data and db_data["documents"]:
                for idx, text in enumerate(db_data["documents"]):
                    metas = db_data["metadatas"]
                    source = metas[idx].get("source", "unknown") if metas else "unknown"
                    doc_id = db_data["ids"][idx] if db_data["ids"] else str(idx)
                    
                    doc_obj = Document(page_content=text, metadata={"source": source})
                    corpus.append({"text": text, "source": source, "id": doc_id})
                    self._corpus_cache.append({
                        "text": text,
                        "source": source,
                        "doc_obj": doc_obj,
                        "id": doc_id
                    })
            
            if corpus:
                self._bm25_instance = SimpleBM25(corpus)
        except Exception:
            self._bm25_instance = None
            self._corpus_cache = []

    def _llm_call(self, prompt: str, temperature: float = 0.0) -> str:
        """Lightweight LLM call for routing and expansion."""
        llm = ChatGoogleGenerativeAI(
            model=config._config_state.get("llm_model", "gemini-2.5-flash"),
            temperature=temperature,
            max_output_tokens=64,
            google_api_key=config.GOOGLE_API_KEY,
        )
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error: {e}"

    def _bm25_search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Score all documents using BM25 and return top K."""
        self._build_bm25_index()
        if not self._bm25_instance or not self._corpus_cache:
            return []
        
        scores = self._bm25_instance.score(query)
        scored_docs = []
        for idx, score in enumerate(scores):
            if score > 0.0: # Only keep documents with keyword matches
                scored_docs.append((self._corpus_cache[idx], score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def ask(self, question: str, history: List[Tuple[str, str]] = None) -> dict:
        """
        Runs the RAG pipeline using configured dynamic retrieval strategies.
        """
        # Load hyperparams
        top_k = config._config_state["retrieval_top_k"]
        temperature = config._config_state["llm_temperature"]
        model_name = config._config_state["llm_model"]
        max_tokens = config._config_state["llm_max_tokens"]
        system_prompt_template = config._config_state["system_prompt"]
        
        # Advanced switches
        rag_mode = config._config_state.get("rag_mode", "standard") # standard | hybrid | cognitive
        rerank_enabled = config._config_state.get("rerank_enabled", False)
        query_router_enabled = config._config_state.get("query_router_enabled", True)

        start_time = time.time()
        routing_decision = "RAG"
        expanded_queries = []
        hybrid_fusion_log = []
        
        # --- 1. Query Routing ---
        if query_router_enabled or rag_mode == "cognitive":
            route_prompt = f"""You are an intelligent query router. Analyze the input question:
"{question}"

Classify it into exactly one of three categories:
- 'GENERAL': Simple greetings, casual talk, questions about your self identity, or basic questions that do NOT require looking up company documentation.
- 'SUMMARY': Requests to summarize, synthesize, list, or explain all documents in the database.
- 'RAG': Specific factual queries seeking answers from remote policies, stipend guidelines, timesheet processes, etc.

Output ONLY the category name: GENERAL, SUMMARY, or RAG. Do not write anything else.
"""
            route_res = self._llm_call(route_prompt).upper()
            if "GENERAL" in route_res:
                routing_decision = "GENERAL"
            elif "SUMMARY" in route_res:
                routing_decision = "SUMMARY"
            else:
                routing_decision = "RAG"

        # --- 2. Context Retrieval Branching ---
        start_retrieval = time.time()
        docs = []
        retrieved_chunks = []

        if routing_decision == "GENERAL":
            # Bypassed retrieval
            context = "(General conversation - retrieval bypassed)"
            retrieval_latency = time.time() - start_retrieval

        elif routing_decision == "SUMMARY":
            # Retrieve all documents to generate summary
            try:
                db_data = self.vectorstore.get()
                if db_data and "documents" in db_data:
                    for idx, text in enumerate(db_data["documents"][:10]): # cap to 10 chunks to fit context window
                        metas = db_data["metadatas"]
                        source = metas[idx].get("source", "unknown") if metas else "unknown"
                        docs.append(Document(page_content=text, metadata={"source": source}))
            except Exception:
                pass
                
            context = format_docs(docs)
            retrieval_latency = time.time() - start_retrieval
            retrieved_chunks = [{
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "similarity": 99.0
            } for d in docs]

        else: # Standard RAG routing
            # Standard, Hybrid, or Cognitive RAG
            if rag_mode == "cognitive":
                # Multi-Query Expansion
                expand_prompt = f"""Generate exactly 3 alternative search queries for the user question. Focus on keywords, synonyms, and different phrasing.
Return them as a numbered list, one per line. Do not write any introduction or notes.

Question: "{question}"
"""
                expand_res = self._llm_call(expand_prompt)
                for line in expand_res.split("\n"):
                    line_cleaned = re.sub(r'^\d+[\.\)\s]+', '', line).strip()
                    if line_cleaned:
                        expanded_queries.append(line_cleaned)

                # Fetch documents for original + expanded queries
                queries_to_run = [question] + expanded_queries[:3]
                all_docs_and_scores = []
                seen_texts = set()

                for q in queries_to_run:
                    try:
                        q_res = self.vectorstore.similarity_search_with_score(q, k=top_k)
                        for doc, score in q_res:
                            doc_key = (doc.page_content, doc.metadata.get("source", "unknown"))
                            if doc_key not in seen_texts:
                                seen_texts.add(doc_key)
                                all_docs_and_scores.append((doc, score))
                    except Exception:
                        pass
                
                # Sort combined results by vector distance
                all_docs_and_scores.sort(key=lambda x: x[1])
                selected = all_docs_and_scores[:top_k]
                
                docs = [d for d, _ in selected]
                for d, score in selected:
                    similarity = max(10.0, min(99.0, (1.0 - (score / 1.6)) * 100.0))
                    retrieved_chunks.append({
                        "text": d.page_content,
                        "source": d.metadata.get("source", "unknown"),
                        "similarity": round(similarity, 1)
                    })
                context = format_docs(docs)
                retrieval_latency = time.time() - start_retrieval

            elif rag_mode == "hybrid":
                # Dense Vector Search + Sparse BM25 Keyword Search + Reciprocal Rank Fusion (RRF)
                n_fusion = top_k * 3
                k_rrf = 60.0
                
                # 1. Vector Rank Map
                try:
                    v_res = self.vectorstore.similarity_search_with_score(question, k=n_fusion)
                except Exception:
                    v_res = []
                
                v_ranks = {}
                for idx, (doc, score) in enumerate(v_res):
                    key = (doc.page_content, doc.metadata.get("source", "unknown"))
                    v_ranks[key] = (idx + 1, score)

                # 2. BM25 Rank Map
                b_res = self._bm25_search(question, k=n_fusion)
                b_ranks = {}
                for idx, (doc_dict, score) in enumerate(b_res):
                    key = (doc_dict["text"], doc_dict["source"])
                    b_ranks[key] = (idx + 1, score)

                # 3. Apply RRF Rank Score calculation
                all_keys = set(v_ranks.keys()).union(set(b_ranks.keys()))
                rrf_list = []
                for key in all_keys:
                    vr, v_score = v_ranks.get(key, (None, None))
                    br, b_score = b_ranks.get(key, (None, None))
                    
                    rrf_score = 0.0
                    if vr is not None:
                        rrf_score += 1.0 / (k_rrf + vr)
                    if br is not None:
                        rrf_score += 1.0 / (k_rrf + br)
                        
                    doc_text, doc_source = key
                    doc_obj = Document(page_content=doc_text, metadata={"source": doc_source})
                    
                    # Convert score to similarity index
                    if v_score is not None:
                        similarity = max(10.0, min(99.0, (1.0 - (v_score / 1.6)) * 100.0))
                    else:
                        similarity = min(90.0, max(20.0, 30.0 + b_score * 5.0))
                        
                    rrf_list.append({
                        "doc_obj": doc_obj,
                        "text": doc_text,
                        "source": doc_source,
                        "vector_rank": vr or "N/A",
                        "bm25_rank": br or "N/A",
                        "rrf_score": rrf_score,
                        "similarity": round(similarity, 1)
                    })

                # Sort by RRF score descending
                rrf_list.sort(key=lambda x: x["rrf_score"], reverse=True)
                selected_rrf = rrf_list[:top_k]
                
                docs = [item["doc_obj"] for item in selected_rrf]
                for item in selected_rrf:
                    retrieved_chunks.append({
                        "text": item["text"],
                        "source": item["source"],
                        "similarity": item["similarity"],
                        "vector_rank": item["vector_rank"],
                        "bm25_rank": item["bm25_rank"],
                        "rrf_score": round(item["rrf_score"], 5)
                    })
                    # Log for UI hybrid rank details
                    hybrid_fusion_log.append({
                        "source": item["source"],
                        "vector_rank": item["vector_rank"],
                        "bm25_rank": item["bm25_rank"],
                        "rrf_score": round(item["rrf_score"], 5)
                    })
                context = format_docs(docs)
                retrieval_latency = time.time() - start_retrieval

            else:
                # Standard RAG
                try:
                    v_res = self.vectorstore.similarity_search_with_score(question, k=top_k)
                except Exception:
                    v_res = []
                
                docs = [d for d, _ in v_res]
                for d, score in v_res:
                    similarity = max(10.0, min(99.0, (1.0 - (score / 1.6)) * 100.0))
                    retrieved_chunks.append({
                        "text": d.page_content,
                        "source": d.metadata.get("source", "unknown"),
                        "similarity": round(similarity, 1)
                    })
                context = format_docs(docs)
                retrieval_latency = time.time() - start_retrieval

        # --- 3. Keyword-Reweighted Reranker ---
        if rerank_enabled and routing_decision == "RAG" and retrieved_chunks:
            # Score chunks based on exact keyword match overlap with query
            q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower())) # ignore words <= 2 chars
            for chunk in retrieved_chunks:
                c_tokens = set(re.findall(r'\b\w{3,}\b', chunk["text"].lower()))
                overlap = q_tokens.intersection(c_tokens)
                # Boost similarity by 3.5% per keyword overlap, cap at 99.5
                chunk["similarity"] = min(99.5, chunk["similarity"] + len(overlap) * 3.5)
            
            # Sort retrieved chunks by boosted similarity descending
            retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            # Reorder Document objects in sync
            reordered_docs = []
            for chunk in retrieved_chunks:
                for d in docs:
                    if d.page_content == chunk["text"]:
                        reordered_docs.append(d)
                        break
            docs = reordered_docs
            context = format_docs(docs)

        # --- 4. Synthesis / Generation ---
        formatted_system = system_prompt_template.format(context=context)
        lc_history = [(role, content) for role, content in (history or [])]

        prompt_messages = [
            ("system", formatted_system),
        ]
        for role, content in lc_history:
            prompt_role = "human" if role == "human" else "ai"
            prompt_messages.append((prompt_role, content))
        prompt_messages.append(("human", question))

        # Format full prompt string for debugger
        formatted_prompt_str = f"SYSTEM PROMPT:\n{formatted_system}\n"
        for role, content in lc_history:
            formatted_prompt_str += f"\n{role.upper()}: {content}"
        formatted_prompt_str += f"\n\nHUMAN: {question}"

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=config.GOOGLE_API_KEY,
        )
        chain = ChatPromptTemplate.from_messages(prompt_messages) | llm | StrOutputParser()

        start_gen = time.time()
        try:
            answer = chain.invoke({})
        except Exception as e:
            answer = f"Synthesis Error: {e}"
        generation_latency = time.time() - start_gen

        # Calculate metrics and RAG Triad scores
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        metrics = calculate_rag_metrics(question, context, answer)
        
        total_latency = time.time() - start_time

        return {
            "answer": answer,
            "sources": sources,
            "diagnostics": {
                "retrieval_latency_ms": int(retrieval_latency * 1000),
                "generation_latency_ms": int(generation_latency * 1000),
                "total_latency_ms": int(total_latency * 1000),
                "retrieved_chunks": retrieved_chunks,
                "formatted_prompt": formatted_prompt_str,
                "metrics": metrics,
                "rag_mode": rag_mode,
                "routing_decision": routing_decision,
                "expanded_queries": expanded_queries,
                "hybrid_fusion_log": hybrid_fusion_log,
                "rerank_applied": rerank_enabled
            }
        }


_pipeline_instance: RAGPipeline = None


def get_pipeline() -> RAGPipeline:
    """Lazily construct a single shared RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
