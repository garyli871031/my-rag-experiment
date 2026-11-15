import os
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
#from langchain_cohere import CohereRerank
from dotenv import load_dotenv

load_dotenv()

chunks = [
    # Tesla - Financial & Production
    "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
    "Tesla's automotive gross margin improved to 19.3% this quarter.",
    "Tesla Cybertruck production ramp begins in 2024 with initial deliveries.",
    "Tesla announced plans to expand Gigafactory production capacity.",
    "Tesla stock price reached new highs following earnings announcement.",
    "Tesla's energy storage business grew 40% year-over-year.",
    "Tesla continues to lead in electric vehicle market share globally.",
    "Tesla Model Y became the best-selling vehicle worldwide.",
    "Tesla reported strong free cash flow generation of $7.5 billion.",
    "Tesla's Full Self-Driving revenue increased significantly.",
    
    # Microsoft - Development & Acquisitions
    "Microsoft acquired GitHub for $7.5 billion in 2018.",
    "Microsoft's cloud revenue Azure grew 29% year-over-year.",
    "Microsoft announced new AI features for Visual Studio Code.",
    "Microsoft Teams integration with GitHub enhances developer workflow.",
    "Microsoft's developer tools division sees strong adoption.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",
    "Microsoft's productivity suite gained 50 million new users.",
    "Microsoft announced new Surface devices for developers.",
    "Microsoft's AI Copilot features expand to more development tools.",
    "Microsoft's enterprise solutions drive revenue growth.",
    
    # NVIDIA - AI & Hardware
    "NVIDIA's data center revenue reached $47.5 billion annually.",
    "NVIDIA's H100 GPUs see unprecedented demand for AI training.",
    "NVIDIA announced next-generation Blackwell architecture.",
    "NVIDIA's gaming revenue declined due to crypto market changes.",
    "NVIDIA's automotive AI platform partnerships expanded.",
    "NVIDIA's AI chip shortage affects cloud providers.",
    "NVIDIA stock valuation exceeds $2 trillion market cap.",
    "NVIDIA's CUDA platform dominates AI development.",
    "NVIDIA announced new AI inference chips for edge computing.",
    "NVIDIA's partnership with major cloud providers strengthens.",
    
    # Google/Alphabet - AI & Cloud
    "Google's AI investments total over $100 billion in recent years.",
    "Google Cloud revenue grew 35% reaching $8.4 billion quarterly.",
    "Google announced Gemini AI model competing with GPT-4.",
    "Google's search advertising revenue remains strong at $59 billion.",
    "Google's Workspace products integrate advanced AI features.",
    "Google announced quantum computing breakthroughs.",
    "Google's autonomous vehicle division Waymo expands operations.",
    "Google's AI research published breakthrough papers.",
    "Google's cloud AI services see enterprise adoption.",
    "Google faces regulatory scrutiny over AI dominance.",
    
    # Noisy/Less Relevant Chunks
    "The Tesla coil was invented by Nikola Tesla in 1891.",
    "Microsoft Excel spreadsheet formulas can be complex for beginners.",
    "NVIDIA Shield TV streaming device gets software update.",
    "Google Maps navigation improved with real-time traffic data.",
    "Production delays affected multiple manufacturing sectors.",
    "Financial markets showed volatility during earnings season.",
    "Revenue recognition standards changed for software companies.",
    "Hardware components face supply chain constraints globally.",
    "Development tools market grows with remote work trends.",
    "AI research requires significant computational resources.",
    "Quarterly reports show mixed results across tech sector.",
    "Stock market analysts upgrade technology sector ratings.",
    "Cloud computing adoption accelerates in enterprise market.",
    "Data center construction increases globally.",
    "Semiconductor shortage impacts various industries.",
    "Electric vehicle charging infrastructure expands rapidly.",
    "Software development productivity tools gain popularity.",
    "Machine learning frameworks become more accessible.",
    "Enterprise software licensing models evolve.",
    "Technology conferences showcase latest innovations."
]

print(f"Created {len(chunks)} sample chunks for demonstration")


# Convert to Document objects
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

embedding_model = OllamaEmbeddings(model="embeddinggemma-2")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})


# 2. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

print("Setup complete!\n")


query = "Tesla financial performance and production updates"

print("STEP 1: Hybrid Search Results")
print("-"*50)

retrieved_docs = hybrid_retriever.invoke(query)  # Get top 25 for reranking

# Show top 10 from hybrid search
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print(f"\n(Retrieved {len(retrieved_docs)} total chunks for reranking)\n")


from sentence_transformers import CrossEncoder

# Load a lightweight, highly effective local cross-encoder model
# (It downloads automatically to your cache on first run)
reranker_model = CrossEncoder("BAAI/bge-reranker-large")
# Or use a faster, lighter alternative: "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(query: str, retrieved_docs: list, top_n: int = 3):
  """Re-ranks retrieved documents using a local cross-encoder model."""
  if not retrieved_docs:
    return []

  # Prepare pairs of [query, document_content]
  pairs = [[query, doc.page_content] for doc in retrieved_docs]

  # Predict relevance scores
  scores = reranker_model.predict(pairs)

  # Attach scores and sort
  scored_docs = list(zip(retrieved_docs, scores))
  scored_docs.sort(key=lambda x: x[1], reverse=True)

  # Return top_n documents
  return [doc for doc, score in scored_docs[:top_n]]


# --- Example Usage in your RAG pipeline ---
# query = "What are the configuration steps for RAG ingestion?"
# raw_docs = vector_store.similarity_search(query, k=10)
# final_docs = rerank_documents(query, raw_docs, top_n=3)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

print("STEP 2: After Local Cross-Encoder Reranking (Top 10)")
print("-"*50)

# 1. Initialize the underlying local cross-encoder model
# (BGE reranker or a faster MS-MARCO model will download locally on first run)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
# Alternative lightweight/fast option: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 2. Wrap it in LangChain's CrossEncoderReranker document compressor
reranker = CrossEncoderReranker(model=model, top_n=10)

# Rerank the retrieved documents
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# Show reranked results
for i, doc in enumerate(reranked_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print("\n" + "="*80)
print("ANALYSIS:")
print("✅ Hybrid Search: Mixed relevant and irrelevant results")
print("✅ Local Reranking: Most relevant Tesla financial/production info at top")
print("✅ Notice how reranking moved the most contextually relevant chunks higher")

# Optional: Show the difference more clearly
print("\n" + "="*80)
print("KEY IMPROVEMENTS AFTER RERANKING:")
print("-"*40)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

print("\n" + "="*80)
print("FINAL: RAG with Reranked Context")
print("-"*40)

# Use top 5 reranked documents for final answer
top_reranked = reranked_docs[:5]

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in top_reranked])}

Please provide a clear, helpful answer using only the information from these documents."""

model = ChatOllama(model="qwen2.5-coder:7b", temperature=0.2)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
print("Generated Response:")
print(result.content)
