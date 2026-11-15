import os
from unstructured.partition.auto import partition
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


class UnifiedRAGService:
    """
    Unified RAG backend combining:
    - Ingestion (from ingest.py): Local unstructured partitioning for PDFs, images, and text.
    - Retrieval (from ragv1.py): PGVector dense retrieval + BAAI/bge-reranker-large cross-encoder[cite: 1].
    """

    def __init__(self, connection_string: str, top_k_initial: int = 15, top_n_final: int = 5):
        self.top_k_initial = top_k_initial
        self.top_n_final = top_n_final
        self.collection_name = "chatvectorv1_collection"

        # 1. Initialize shared local embedding model (all-MiniLM-L6-v2)
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Initialize local Cross-Encoder ONCE so queries stay fast
        print("Loading local cross-encoder model (BAAI/bge-reranker-large)...")
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-large")

        # 3. Initialize PGVector connection
        self.connection_string = None
        self.vector_store = None
        self.vector_retriever = None
        self.reset_connection(connection_string)

    @staticmethod
    def _normalize_sa_uri(uri: str) -> str:
        """Converts postgresql:// to postgresql+psycopg:// for SQLAlchemy/PGVector compatibility."""
        if uri.startswith("postgresql://"):
            return uri.replace("postgresql://", "postgresql+psycopg://", 1)
        if uri.startswith("postgres://"):
            return uri.replace("postgres://", "postgresql+psycopg://", 1)
        return uri

    def reset_connection(self, connection_string: str):
        """Called by app.py on startup and whenever the Postgres URI changes in the sidebar."""
        if not connection_string:
            return
        sa_uri = self._normalize_sa_uri(connection_string)
        if sa_uri == self.connection_string and self.vector_store is not None:
            return

        self.connection_string = sa_uri
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k_initial}
        )

    # =========================================================================
    # INGESTION & EXTRACTION (from ingest.py)
    # =========================================================================

    def _partition_file(self, file_path: str) -> list[Document]:
        """Runs local unstructured layout analysis and returns LangChain Documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        # partition_via_api=False ensures local execution using poppler & tesseract
        elements = partition(filename=file_path, partition_via_api=False)
        docs = []
        source_name = os.path.basename(file_path)

        for el in elements:
            text = str(el).strip()
            if not text:
                continue

            page_num = (
                el.metadata.page_number
                if hasattr(el, "metadata") and el.metadata.page_number
                else 1
            )
            category = getattr(el, "category", "NarrativeText")

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "element_type": category,
                    },
                )
            )
        return docs

    def extract_text(self, file_path: str) -> str:
        """Used by app.py's 'Analyze with AI' sidebar button."""
        docs = self._partition_file(file_path)
        if not docs:
            return "No readable text could be extracted from this document."
        return "\n\n".join(
            [f"[Page {d.metadata['page']} | {d.metadata['element_type']}]\n{d.page_content}" for d in docs]
        )

    def ingest_pdf(self, file_path: str) -> str:
        """Ingests a PDF into chatvectorv1 using local unstructured layout parsing."""
        if self.vector_store is None:
            return "Error: Vector store is not initialized."
        try:
            docs = self._partition_file(file_path)
            if not docs:
                return f"No text chunks extracted from {os.path.basename(file_path)}."
            self.vector_store.add_documents(docs)
            return f"Successfully ingested {len(docs)} chunks from '{os.path.basename(file_path)}' into {self.collection_name}."
        except Exception as e:
            return f"Error ingesting PDF: {str(e)}"

    def ingest_image(self, file_path: str) -> str:
        """Ingests an image (via local Tesseract OCR in unstructured) into chatvectorv1."""
        if self.vector_store is None:
            return "Error: Vector store is not initialized."
        try:
            docs = self._partition_file(file_path)
            if not docs:
                return f"No OCR text extracted from image '{os.path.basename(file_path)}'."
            self.vector_store.add_documents(docs)
            return f"Successfully ingested {len(docs)} OCR chunks from '{os.path.basename(file_path)}' into {self.collection_name}."
        except Exception as e:
            return f"Error ingesting image: {str(e)}"

    def ingest_text(self, text: str, source: str = "manual_text") -> str:
        """Ingests raw text or .txt files into chatvectorv1."""
        if self.vector_store is None:
            return "Error: Vector store is not initialized."
        try:
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": source, "page": 1, "element_type": "NarrativeText"},
                )
                for chunk in chunks
            ]
            if not docs:
                return f"No valid text found in '{source}'."
            self.vector_store.add_documents(docs)
            return f"Successfully ingested {len(docs)} text chunks from '{source}' into {self.collection_name}."
        except Exception as e:
            return f"Error ingesting text: {str(e)}"

    # =========================================================================
    # RETRIEVAL & RERANKING (from ragv1.py)
    # =========================================================================

    def query(self, query: str) -> str:
        """Executes PGVector dense search followed by BAAI/bge-reranker-large cross-encoder reranking."""
        if self.vector_retriever is None:
            return "Error: Vector store is not initialized."

        try:
            print(f"[Hybrid Rerank Retriever] Running query: '{query}'")
            retrieved_docs = self.vector_retriever.invoke(query)

            if not retrieved_docs:
                return "No relevant information found in the knowledge base."

            print(f"Retrieved {len(retrieved_docs)} candidate chunks. Applying cross-encoder...")
            pairs = [[query, doc.page_content] for doc in retrieved_docs]
            scores = self.reranker_model.predict(pairs)

            scored_docs = sorted(
                zip(retrieved_docs, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            final_docs = [doc for doc, _ in scored_docs[:self.top_n_final]]
            print(f"Selected top {len(final_docs)} chunks after reranking.")

            context = "\n\n".join(
                [
                    f"- [Source: {d.metadata.get('source', 'Unknown')}, Page {d.metadata.get('page', 1)}] {d.page_content}"
                    for d in final_docs
                ]
            )
            return f"Retrieved Context (Reranked):\n{context}"

        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"


# Singleton instance imported by app.py (`from unified_rag_service import rag_service`)
DEFAULT_DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:871031@localhost:5432/chatvectorv1"
)
rag_service = UnifiedRAGService(connection_string=DEFAULT_DB_URI)