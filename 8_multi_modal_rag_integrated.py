import os
import json
from typing import List
from dotenv import load_dotenv

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

load_dotenv()

class AgenticChunker:
    def __init__(self, model_name="gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def chunk(self, text: str) -> List[str]:
        if len(text) < 500:
            return [text]
        try:
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
        except Exception as e:
            print(f"     ⚠️ Agentic chunking failed: {e}. Falling back to simple split.")
            return [text[i:i+500] for i in range(0, len(text), 500)]

def partition_document(file_path: str):
    print(f"📄 Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"✅ Extracted {len(elements)} elements")
    return elements

def create_chunks_hybrid(elements, agentic_chunker=None):
    print("🔨 Creating smart chunks (Title + Agentic)...")
    initial_chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    if not agentic_chunker:
        return initial_chunks

    refined_chunks = []
    for chunk in initial_chunks:
        if len(chunk.text) > 800 and not any(type(el).__name__ in ['Table', 'Image'] for el in chunk.metadata.orig_elements):
            sub_chunks = agentic_chunker.chunk(chunk.text)
            for sc in sub_chunks:
                new_doc = Document(page_content=sc, metadata=chunk.metadata.to_dict())
                refined_chunks.append(new_doc)
        else:
            refined_chunks.append(Document(page_content=chunk.text, metadata=chunk.metadata.to_dict()))
            
    print(f"✅ Created {len(refined_chunks)} refined chunks")
    return refined_chunks

def separate_content_types(chunk):
    content_data = {
        'text': chunk.page_content if isinstance(chunk, Document) else chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    if hasattr(chunk, 'metadata') and 'orig_elements' in chunk.metadata:
        for element in chunk.metadata['orig_elements']:
            element_type = type(element).__name__
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        prompt_text = (
            "You are creating a searchable description for document content retrieval.\n\n"
            "CONTENT TO ANALYZE:\n"
            "TEXT CONTENT:\n"
            f"{text}\n"
        )
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += (
            "\nYOUR TASK:\n"
            "Generate a comprehensive, searchable description that covers:\n"
            "1. Key facts, numbers, and data points from text and tables\n"
            "2. Main topics and concepts discussed\n"
            "3. Questions this content could answer\n"
            "4. Visual content analysis (charts, diagrams, patterns in images)\n"
            "5. Alternative search terms users might use\n\n"
            "Make it detailed and searchable - prioritize findability over brevity.\n"
            "SEARCHABLE DESCRIPTION:"
        )

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        return text[:300]

def summarise_chunks(chunks):
    print("🧠 Processing chunks with AI Summaries...")
    langchain_documents = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{total_chunks}")
        content_data = separate_content_types(chunk)
        if content_data['tables'] or content_data['images']:
            enhanced_content = create_ai_enhanced_summary(
                content_data['text'],
                content_data['tables'], 
                content_data['images']
            )
        else:
            enhanced_content = content_data['text']
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                }),
                "orig_elements": chunk.metadata.get('orig_elements', []) if hasattr(chunk, 'metadata') else []
            }
        )
        langchain_documents.append(doc)
    return langchain_documents

class MultiModalRAGSystem:
    def __init__(self, persist_directory="dbv2/chroma_db"):
        self.persist_dir = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=5)
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def ingest(self, documents: List[Document]):
        print("🔮 Ingesting into Hybrid Store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 15
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        print("✅ Hybrid Store setup complete.")

    def retrieve(self, query: str):
        print(f"🔍 Retrieving for: {query}")
        raw_docs = self.hybrid_retriever.invoke(query)
        reranked_docs = self.reranker.compress_documents(raw_docs, query)
        return reranked_docs

    def answer(self, query: str):
        chunks = self.retrieve(query)
        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n"
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            prompt_text += "\n"
        
        prompt_text += (
            "Please provide a clear, comprehensive answer using the text, tables, and images above. "
            "If the documents don't contain sufficient information, say you don't have enough information.\n"
            "ANSWER:"
        )
        
        message_content = [{"type": "text", "text": prompt_text}]
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                for img in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    })
        
        return self.llm.invoke([HumanMessage(content=message_content)]).content

def run_complete_pipeline(pdf_path: str):
    print("🚀 Starting Integrated Multi-Modal Pipeline")
    elements = partition_document(pdf_path)
    chunker = AgenticChunker()
    chunks = create_chunks_hybrid(elements, agentic_chunker=chunker)
    summarised_docs = summarise_chunks(chunks)
    system = MultiModalRAGSystem()
    system.ingest(summarised_docs)
    return system

if __name__ == "__main__":
    pdf_path = "/home/gary/projects/rag-for-beginners/docs/attention-is-all-you-need.pdf"
    rag_system = run_complete_pipeline(pdf_path)
    query = "How many attention heads does the Transformer use, and what is the dimension of each head?"
    print(f"\nQuery: {query}")
    print("\n--- Final Answer ---")
    print(rag_system.answer(query))
