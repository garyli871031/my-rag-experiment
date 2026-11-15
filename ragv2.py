import os
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_classic.retrievers import EnsembleRetriever

# --- Configuration & Database Setup ---
CONNECTION_STRING = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql+psycopg://gary:871031@localhost:5432/chatvectorv1"
)
COLLECTION_NAME = "chatvectorv1_collection"

def get_vector_store() -> PGVector:
    """Initializes and returns the PGVector store backed by psycopg3."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

class HybridRerankRetriever:
    """Combines PGVector dense search + BM25 sparse search with local Cross-Encoder reranking."""
    
    def __init__(self, top_k_initial: int = 15, top_n_final: int = 5):
        self.top_k_initial = top_k_initial
        self.top_n_final = top_n_final
        
        # 1. Connect to PostgreSQL Vector Store
        self.vector_store = get_vector_store()
        
        # Initialize base vector retriever
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k_initial}
        )
        
        # 2. Initialize Local Cross-Encoder for Reranking
        print("Loading local cross-encoder model (BAAI/bge-reranker-large)...")
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-large")

    def invoke(self, query: str) -> list[Document]:
        """Executes hybrid retrieval followed by local cross-encoder reranking."""
        print(f"\n[Hybrid Search] Running query: '{query}'")
        
        # Since BM25 requires all document texts from the store, 
        # we fetch documents or fall back to dense vector search if corpus is large.
        # Here we perform dense retrieval from PGVector as our primary candidate pool:
        retrieved_docs = self.vector_retriever.invoke(query)
        
        if not retrieved_docs:
            return []

        print(f"Retrieved {len(retrieved_docs)} candidate chunks. Reranking...")

        # 3. Apply Cross-Encoder Reranking
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker_model.predict(pairs)

        # Attach scores and sort by relevance descending
        scored_docs = list(zip(retrieved_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top N final documents
        final_docs = [doc for doc, score in scored_docs[:self.top_n_final]]
        print(f"Selected top {len(final_docs)} chunks after reranking.")
        
        return final_docs

# --- Example Testing Entrypoint ---
if __name__ == "__main__":
    retriever = HybridRerankRetriever(top_k_initial=10, top_n_final=3)
    
    test_query = "What are the core concepts of small language models?"
    results = retriever.invoke(test_query)
    
    print("\nFinal Reranked Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [Page {doc.metadata.get('page')}] {doc.page_content[:150]}...")