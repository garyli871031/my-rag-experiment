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

# Google AI Models
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class AgenticChunker:
    """Uses an LLM to split text into logical, semantically coherent chunks."""
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
    """A comprehensive RAG system combining Hybrid Search, Reranking, and Multi-Modal support with PostgreSQL."""
    def __init__(self, collection_name="unified_rag_collection"):
        # DB Connection
        self.connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:871031@localhost:5432/chatvectordb")
        self.collection_name = collection_name
        
        # Models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        self.chunker = AgenticChunker()
        
        # Reranker
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=5)
        
        # State
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.documents = []

    def _process_text_chunk(self, text: str, metadata: dict) -> List[Document]:
        sub_texts = self.chunker.chunk(text)
        return [Document(page_content=st, metadata=metadata) for st in sub_texts]

    def load_text(self, text: str, source: str = "text_input"):
        print(f"📝 Loading text from {source}...")
        docs = self._process_text_chunk(text, {"source": source})
        self.documents.extend(docs)

    def load_pdf(self, pdf_path: str):
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
                
        self.documents.extend(refined_docs)
        print(f"✅ Loaded {len(refined_docs)} chunks from PDF.")

    def build_index(self):
        if not self.documents:
            print("⚠️ No documents loaded to index.")
            return

        print("🔮 Building Hybrid Index in PostgreSQL...")
        # PGVector does not support lists/dicts in metadata. Filter them out.
        filtered_docs = filter_complex_metadata(self.documents)
        
        self.vectorstore = PGVector.from_documents(
            documents=filtered_docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True
        )
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 15
        
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        print("✅ Hybrid index built successfully in PostgreSQL.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.hybrid_retriever:
            raise ValueError("Index not built. Call build_index() first.")
        
        raw_docs = self.hybrid_retriever.invoke(query)
        reranked_docs = self.reranker.compress_documents(raw_docs, query)
        return reranked_docs[:top_k]

    def answer(self, query: str) -> str:
        docs = self.retrieve(query)
        
        context_parts = []
        image_contents = []
        
        for i, doc in enumerate(docs):
            part = f"--- Document {i+1} ---\nTEXT: {doc.page_content}"
            
            if 'orig_elements' in doc.metadata:
                for el in doc.metadata['orig_elements']:
                    if type(el).__name__ == 'Table':
                        table_html = getattr(el.metadata, 'text_as_html', el.text)
                        part += f"\nTABLE: {table_html}"
                    elif type(el).__name__ == 'Image':
                        if hasattr(el.metadata, 'image_base64'):
                            image_contents.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{el.metadata.image_base64}"}
                            })
            context_parts.append(part)
            
        prompt_text = (
            f"Based on the following documents, please answer this question: {query}\n\n"
            f"CONTEXT:\n" + "\n\n".join(context_parts) + 
            "\n\nAnswer comprehensively using all provided text and visual data. "
            "If information is missing, say so."
        )
        
        message_content = [{"type": "text", "text": prompt_text}]
        message_content.extend(image_contents)
        
        return self.llm.invoke([HumanMessage(content=message_content)]).content

if __name__ == "__main__":
    rag = UnifiedRAG()
    
    pdf_path = "/home/gary/projects/rag-for-beginners/docs/attention-is-all-you-need.pdf"
    if os.path.exists(pdf_path):
        rag.load_pdf(pdf_path)
    
    sample_text = "Tesla reported record revenue of $25.2B in Q3 2024. The Model Y became the best-selling vehicle globally."
    rag.load_text(sample_text, source="corporate_reports")
    
    rag.build_index()
    
    queries = [
        "How many attention heads does the Transformer use?", 
        "What was Tesla's revenue in Q3 2024?"              
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        print(f"Response: {rag.answer(q)}")
        print("-" * 50)
