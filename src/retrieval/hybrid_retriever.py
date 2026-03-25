"""Hybrid retrieval: BM25 sparse + dense vector with RRF fusion."""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import numpy as np


def reciprocal_rank_fusion(results_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, {"doc": doc, "score": 0})
            scores[key]["score"] += 1 / (k + rank + 1)
    return [v["doc"] for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


class HybridRetriever:
    """Combines BM25 and dense vector retrieval with RRF score fusion."""

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "ragcore",
        k: int = 8,
    ):
        self.k = k
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Chroma.from_documents(documents, self.embeddings, collection_name=collection_name)
        self.bm25 = BM25Retriever.from_documents(documents, k=k)
        self.dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve top-k documents using hybrid search."""
        k = k or self.k
        dense_results = self.dense_retriever.invoke(query)
        sparse_results = self.bm25.invoke(query)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:k]
