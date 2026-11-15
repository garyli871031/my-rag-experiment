import os
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

class HybridRerankRetriever:
    """Combines PGVector dense search with local Cross-Encoder reranking for chatvectorv1."""
    
    def __init__(self, connection_string: str, top_k_initial: int = 15, top_n_final: int = 5):
        self.top_k_initial = top_k_initial
        self.top_n_final = top_n_final
        
        # 1. Initialize PostgreSQL Vector Store connection (chatvectorv1)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="chatvectorv1_collection",
            connection=connection_string,
            use_jsonb=True,
        )
        
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k_initial}
        )
        
        # 2. Initialize Local Cross-Encoder for Reranking
        print("Loading local cross-encoder model (BAAI/bge-reranker-large)...")
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-large")

    def invoke(self, query: str) -> list[Document]:
        """Executes vector retrieval followed by local cross-encoder reranking."""
        print(f"\n[Hybrid Rerank Retriever] Running query: '{query}'")
        
        # Fetch candidate pool from PostgreSQL vector store
        retrieved_docs = self.vector_retriever.invoke(query)
        
        if not retrieved_docs:
            return []

        print(f"Retrieved {len(retrieved_docs)} candidate chunks. Applying cross-encoder...")

        # Apply Cross-Encoder Reranking
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker_model.predict(pairs)

        # Sort by relevance score descending
        scored_docs = list(zip(retrieved_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top N final documents
        final_docs = [doc for doc, score in scored_docs[:self.top_n_final]]
        print(f"Selected top {len(final_docs)} chunks after reranking.")
        
        return final_docs

    @tool
    def query_knowledge_base(query: str) -> str:
    """Queries the multi-modal RAG knowledge base using hybrid search and local reranking."""
    try:
        # Initialize retriever with current active database URI
        retriever = HybridRerankRetriever(connection_string=get_db_uri())
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant information found in the knowledge base."
            
        # Format retrieved chunks for the LLM
        context = "\n\n".join([f"- [Page {d.metadata.get('page', 1)}] {d.page_content}" for d in docs])
        return f"Retrieved Context (Reranked):\n{context}"
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"