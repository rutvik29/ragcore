"""FastAPI RAG service with streaming support."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile, os

app = FastAPI(title="RAGCore API", version="1.0.0")

_retriever = None
_reranker = None


class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    faithfulness_score: Optional[float] = None


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the RAG pipeline."""
    global _retriever
    from src.ingestion.loader import load_document
    from src.chunking.chunker import SemanticChunker
    from src.retrieval.hybrid_retriever import HybridRetriever

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    docs = load_document(tmp_path)
    chunks = SemanticChunker().chunk(docs)
    _retriever = HybridRetriever(chunks)
    os.unlink(tmp_path)
    return {"status": "indexed", "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline."""
    if not _retriever:
        raise HTTPException(400, "No documents indexed. POST to /ingest first.")
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA

    docs = _retriever.retrieve(request.query, request.k)
    if request.rerank:
        from src.reranking.reranker import CrossEncoderReranker
        docs = CrossEncoderReranker().rerank(request.query, docs)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    context = "\n\n".join([d.page_content for d in docs])
    response = llm.invoke(f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer based on context only:")
    return QueryResponse(
        answer=response.content,
        sources=[{"content": d.page_content[:200], "metadata": d.metadata} for d in docs],
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "indexed": _retriever is not None}
