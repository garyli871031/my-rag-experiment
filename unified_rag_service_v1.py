import os
import json
from typing import List, Union, Optional
from dotenv import load_dotenv

# Parsing & Chunking
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain Core & Vector Stores
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores.utils import filter_complex_metadata
import base64
from PIL import Image

# Google AI Models
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class AgenticChunker:
    def __init__(self, model_name="gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def chunk(self, text: str) -> List[str]:
        if len(text) < 500:
            return [text]
        try:
            prompt = f"""You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200-500 characters
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{text}

Return the text with <<<SPLIT>>> markers where you want to split:"""
            response = self.llm.invoke(prompt)
            marked_text = response.content if isinstance(response.content, str) else "".join([str(item) for item in response.content])
            chunks = [c.strip() for c in marked_text.split("<<<SPLIT>>>") if c.strip()]
            return chunks
        except Exception as e:
            print(f"     ⚠️ Agentic chunking failed: {e}. Falling back to simple split.")
            return [text[i:i+500] for i in range(0, len(text), 500)]

class UnifiedRAG:
    def __init__(self, collection_name="unified_rag_collection"):
        self.connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:***@localhost:5432/chatvectordb")
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        self.chunker = AgenticChunker()
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=5)
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def reset_connection(self, new_connection_string: str):
        """Updates the connection string and clears cached stores to force re-initialization."""
        self.connection_string = new_connection_string
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        print(f"Connection reset to: {new_connection_string}")

    def _initialize_store(self):
        if self.vectorstore is None:
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                embeddings=self.embeddings,
                connection=self.connection_string,
                use_jsonb=True
            )
            self._refresh_hybrid_retriever()

    def ingest_pdf(self, pdf_path: str):
        print(f"📄 Partitioning PDF: {pdf_path}...")
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        initial_chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500
        )
        refined_docs = []
        for chunk in initial_chunks:
            meta = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else chunk.metadata
            has_complex = any(type(el).__name__ in ['Table', 'Image'] for el in chunk.metadata.orig_elements)
            if len(chunk.text) > 800 and not has_complex:
                sub_texts = self.chunker.chunk(chunk.text)
                for st in sub_texts:
                    refined_docs.append(Document(page_content=st, metadata=meta))
            else:
                refined_docs.append(Document(page_content=chunk.text, metadata=meta))
        
        self._initialize_store()
        filtered_docs = filter_complex_metadata(refined_docs)
        self.vectorstore.add_documents(filtered_docs)
        self._refresh_hybrid_retriever()
        return f"Successfully ingested {len(refined_docs)} chunks from {os.path.basename(pdf_path)}."

    def ingest_text(self, text: str, source: str = "text_input"):
        docs = self.chunker.chunk(text)
        refined_docs = [Document(page_content=st, metadata={"source": source}) for st in docs]
        self._initialize_store()
        filtered_docs = filter_complex_metadata(refined_docs)
        self.vectorstore.add_documents(filtered_docs)
        self._refresh_hybrid_retriever()
        return f"Successfully ingested {len(refined_docs)} chunks from {source}."

    def ingest_image(self, image_path: str):
        print(f"🖼️ Analyzing image: {image_path}...")
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image in great detail. If there is any text in the image, transcribe it exactly. This description will be used for a searchable knowledge base, so be thorough."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            )
            
            description = self.llm.invoke([message]).content
            return self.ingest_text(description, source=os.path.basename(image_path))
        except Exception as e:
            return f"Failed to ingest image {image_path}: {e}"

    def extract_text(self, file_path: str) -> str:
        """Extracts text from a file (PDF, Text, or Image) without storing it in the database."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                elements = partition_pdf(filename=file_path, strategy="fast")
                return "\n".join([str(el) for el in elements])
            elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic']:
                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Transcribe all text in this image and describe the content."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                )
                return self.llm.invoke([message]).content
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            return f"Error extracting text: {e}"

    def _refresh_hybrid_retriever(self):
        if self.vectorstore is None:
            return

        # 1. Setup Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # 2. Setup BM25 Retriever (Keyword Search)
        try:
            # Fetch all document contents from PGVector using psycopg
            all_docs = []
            import psycopg # Ensure it's available in this scope
            with psycopg.connect(self.connection_string, autocommit=True) as conn:
                with conn.cursor() as cur:
                    # Table name in PGVector is typically 'langchain_pg_embedding'
                    # We fetch the 'document' column which contains the text
                    cur.execute("SELECT document FROM langchain_pg_embedding")
                    for row in cur.fetchall():
                        all_docs.append(Document(page_content=row[0]))
            
            if not all_docs:
                self.hybrid_retriever = vector_retriever
                return

            self.bm25_retriever = BM25Retriever.from_documents(all_docs)
            
            # 3. Combine into Ensemble Retriever
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
        except Exception as e:
            print(f"⚠️ Hybrid retriever setup failed: {e}. Falling back to vector search.")
            self.hybrid_retriever = vector_retriever

    def query(self, query: str) -> str:
        self._initialize_store()
        raw_docs = self.hybrid_retriever.invoke(query) if self.hybrid_retriever else self.vectorstore.similarity_search(query, k=15)
        reranked_docs = self.reranker.compress_documents(raw_docs, query)
        context = "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(reranked_docs)])
        prompt = f"Based on the following context, answer the question: {query}\n\nCONTEXT:\n{context}"
        return self.llm.invoke([HumanMessage(content=prompt)]).content

rag_service = UnifiedRAG()
