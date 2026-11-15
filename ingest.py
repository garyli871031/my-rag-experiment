import os
from unstructured.partition.auto import partition
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

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

def ingest_pdf_local(pdf_path: str):
    """Parses a local PDF using unstructured layout analysis and ingests chunks into chatvectorv1."""
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    print(f"Partitioning document locally: {pdf_path} using unstructured...")
    
    # partition_via_api=False ensures it runs entirely locally using poppler & tesseract
    elements = partition(filename=pdf_path, partition_via_api=False)
    
    docs = []
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
            
        page_num = el.metadata.page_number if hasattr(el, "metadata") and el.metadata.page_number else 1
        category = getattr(el, "category", "NarrativeText")
        
        doc = Document(
            page_content=text,
            metadata={
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "element_type": category
            }
        )
        docs.append(doc)
        
    print(f"Extracted {len(docs)} structured chunks. Connecting to database...")
    
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
    print(f"Successfully ingested {len(docs)} chunks into chatvectorv1!")

if __name__ == "__main__":
    # Target PDF for your ingestion pipeline
    target_pdf = "./docs/attention-is-all-you-need.pdf"
    ingest_pdf_local(target_pdf)