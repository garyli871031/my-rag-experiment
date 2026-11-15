import os
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class AgenticChunker:
    def __init__(self, model_name="gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def chunk(self, text: str) -> List[str]:
        prompt = (
            "You are a text chunking expert. Split this text into logical chunks.\n\n"
            "Rules:\n"
            "- Each chunk should be around 200-500 characters\n"
            "- Split at natural topic boundaries\n"
            "- Keep related information together\n"
            "- Put \"<<<SPLIT>>>\" between chunks\n\n"
            f"Text:\n{text}\n\n"
            "Return the text with <<<SPLIT>>> markers where you want to split:"
        )
        response = self.llm.invoke(prompt)
        marked_text = response.content if isinstance(response.content, str) else "".join([str(item) for item in response.content])
        chunks = [c.strip() for c in marked_text.split("<<<SPLIT>>>") if c.strip()]
        return chunks

class IntegratedRAG:
    def __init__(self, collection_name="integrated_rag_collection"):
        self.embeddings = OllamaEmbeddings(model="embeddinggemma-2")
        self.llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
        
        # Vector Store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Reranker
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=5)
        
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def ingest(self, texts: List[str], metadata: List[dict] = None):
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadata)]
        
        # Add to Vector Store
        self.vectorstore.add_documents(documents)
        
        # Setup BM25
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 15
        
        # Setup Hybrid Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        print(f"Ingested {len(documents)} chunks successfully.")

    def retrieve(self, query: str, top_k: int = 5):
        if not self.hybrid_retriever:
            raise ValueError("No documents ingested yet.")
        
        # 1. Hybrid Search
        raw_docs = self.hybrid_retriever.invoke(query)
        
        # 2. Reranking
        reranked_docs = self.reranker.compress_documents(raw_docs, query)
        
        return reranked_docs[:top_k]

    def answer(self, query: str):
        docs = self.retrieve(query)
        
        context = "\n".join([f"- {doc.page_content}" for doc in docs])
        combined_input = (
            f"Based on the following documents, please answer this question: {query}\n\n"
            f"Documents:\n{context}\n\n"
            "Please provide a clear, helpful answer using only the information from these documents. "
            "If you can't find the answer, say you don't have enough information."
        )
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]
        
        return self.llm.invoke(messages).content

if __name__ == "__main__":
    # 1. Setup
    chunker = AgenticChunker()
    rag = IntegratedRAG()
    
    # 2. Load and Chunk Data
    print("Reading documents...")
    docs_dir = "/home/gary/projects/rag-for-beginners/docs"
    all_chunks = []
    all_meta = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r") as f:
                text = f.read()
                chunks = chunker.chunk(text)
                all_chunks.extend(chunks)
                all_meta.extend([{"source": filename} for _ in chunks])
    
    # 3. Ingest
    print("Ingesting chunks...")
    rag.ingest(all_chunks, all_meta)
    
    # 4. Query
    query = "What are the key points about Tesla and SpaceX?"
    print(f"\nQuery: {query}")
    
    response = rag.answer(query)
    print("\n--- Response ---")
    print(response)
