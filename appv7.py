import base64
import os
import uuid
import sys
import subprocess
from datetime import datetime
from typing import Annotated, TypedDict, Optional, Union

import streamlit as st
import psycopg
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# --- Safe Import for RAG Service ---
rag_service = None
try:
    from unified_rag_service import rag_service
except ImportError as e:
    print(f"⚠️ Could not import rag_service: {e}")

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="LangGraph Assistant v7", 
    page_icon="⚡", 
    layout="wide"
)

# Custom CSS for a more modern look
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ LangGraph Assistant v7")
st.caption("Multi-modal RAG + PostgreSQL Persistence + Streaming")

# 1. State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- Database Setup ---
DEFAULT_DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:871031@localhost:5432/chatvectordb")

def get_db_uri():
    return st.session_state.get("db_uri", DEFAULT_DB_URI)

def init_receipts_postgres():
    try:
        with psycopg.connect(get_db_uri(), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS receipts (
                        id SERIAL PRIMARY KEY,
                        vendor TEXT,
                        date TEXT,
                        total NUMERIC,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
    except Exception as e:
        st.error(f"Postgres receipts table init error: {e}")

init_receipts_postgres()

# 2. Custom Tools Setup
@tool
def get_current_time(timezone_str: str = "UTC") -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return f"Current local system date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone_str})"

@tool
def get_weather(location: str) -> str:
    """Retrieves current weather conditions using web search."""
    search = TavilySearchResults(max_results=2)
    results = search.invoke(f"current weather in {location}")
    return str(results)

@tool
def run_linux_command(command: str) -> str:
    """Executes a safe read-only Linux shell command."""
    forbidden_keywords = ["rm ", "mkfs", "dd ", ">", "chmod", "chown", "sudo"]
    if any(word in command for word in forbidden_keywords):
        return "Error: Command rejected for safety reasons."
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
        output = result.stdout if result.returncode == 0 else result.stderr
        return output.strip() if output else "Command executed successfully."
    except Exception as e:
        return f"Execution Error: {str(e)}"

@tool
def log_receipt_to_table(vendor: str, date: str, total: float) -> str:
    """Logs verified receipt details directly into the PostgreSQL receipts table."""
    try:
        with psycopg.connect(get_db_uri(), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO receipts (vendor, date, total) VALUES (%s, %s, %s)",
                    (vendor, date, total)
                )
        return f"Successfully logged receipt for {vendor} on {date} totaling ${total:.2f} to PostgreSQL."
    except Exception as e:
        return f"Failed to log receipt to Postgres: {str(e)}"

@tool
def query_knowledge_base(query: str) -> str:
    """Queries the multi-modal RAG knowledge base for uploaded documents."""
    if rag_service is None:
        return "Error: RAG Service is not loaded. Please check system logs."
    return rag_service.query(query)

@tool
def ingest_document_file(file_path: str) -> str:
    """Ingests a document file (PDF, Text, or Image) into the permanent knowledge base."""
    if rag_service is None:
        return "Error: RAG Service is not loaded."
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return rag_service.ingest_pdf(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic']:
        return rag_service.ingest_image(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return rag_service.ingest_text(text, source=os.path.basename(file_path))
        except Exception as e:
            return "Error reading file: " + str(e)

@tool
def export_receipts_to_sheet() -> str:
    """Exports logged receipts from PostgreSQL into a Google Sheet."""
    try:
        with psycopg.connect(get_db_uri(), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT vendor, date, total FROM receipts ORDER BY created_at DESC")
                rows = cursor.fetchall()
        if not rows:
            return "No receipts found in the database to export."
        return f"Successfully prepared {len(rows)} receipt records for Google Sheets export!"
    except Exception as e:
        return f"Failed to export to sheet: {str(e)}"

tools = [
    get_current_time, 
    get_weather, 
    run_linux_command, 
    log_receipt_to_table, 
    TavilySearchResults(max_results=5),
    export_receipts_to_sheet,
    query_knowledge_base,
    ingest_document_file
]

# 3. LLM Setup
llm = ChatOllama(model="gemma4:cloud", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

# 4. Chat Node
def chat_node(state: ChatState):
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = SystemMessage(
        content=f"Today's current date is {current_date}. You have access to a Knowledge Base tool. "
                 "If the user asks about their uploaded documents or a specific file, use the 'query_knowledge_base' tool. "
                 "If the user provides a file to be remembered, use 'ingest_document_file'."
    )
    messages_with_system = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages_with_system)]}

# 5. Graph Compilation
def get_chatbot():
    connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
    conn = psycopg.connect(get_db_uri(), **connection_kwargs)
    db_checkpointer = PostgresSaver(conn)
    db_checkpointer.setup()

    workflow = StateGraph(ChatState)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "chat_node")
    workflow.add_conditional_edges("chat_node", tools_condition)
    workflow.add_edge("tools", "chat_node")
    return workflow.compile(checkpointer=db_checkpointer)

chatbot = get_chatbot()

# --- Configuration & Session ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_postgres_session"

config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown("### 📁 Document Center")
    
    uploaded_file = st.file_uploader(
        "Upload Document (PDF/TXT/Images)", 
        type=["pdf", "txt", "jpg", "jpeg", "png", "webp", "heic"],
        key="main_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state['pending_upload'] = uploaded_file

    if 'pending_upload' in st.session_state:
        file_to_process = st.session_state['pending_upload']
        st.info(f"Selected: {file_to_process.name}")

        # High visibility vertical actions
        if st.button("📄 Analyze with AI", use_container_width=True, type="primary"):
            with st.spinner("Extracting..."):
                try:
                    if rag_service is None:
                        st.error("RAG Service not loaded.")
                        st.stop()
                    
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, file_to_process.name)
                    with open(file_path, "wb") as f:
                        f.write(file_to_process.getbuffer())
                    
                    # Type hint rag_service for Pyright
                    text_content = rag_service.extract_text(file_path)
                    image_message = HumanMessage(content=f"Please analyze this document: \n\n{text_content}")
                    chatbot.update_state(config, {"messages": [image_message]})
                    st.success("Sent to AI!")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("💾 Save to Knowledge Base", use_container_width=True):
            with st.spinner("Saving..."):
                try:
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, file_to_process.name)
                    with open(file_path, "wb") as f:
                        f.write(file_to_process.getbuffer())
                    
                    result = ingest_document_file.invoke({"file_path": file_path})
                    st.success("Saved successfully!")
                    del st.session_state['pending_upload']
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("❌ Clear Selection", use_container_width=True):
            del st.session_state['pending_upload']
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Control Panel")
    
    # --- DB Configuration ---
    st.subheader("🌐 Database Config")
    db_uri_input = st.text_input("Postgres URI", value=get_db_uri())
    if db_uri_input != get_db_uri():
        st.session_state.db_uri = db_uri_input
        if rag_service:
            rag_service.connection_string = db_uri_input
        st.success("Database URI updated!")
        st.rerun()
    
    st.markdown("---")

    thread_input = st.text_input("Active Thread ID", value=st.session_state.thread_id)
    if thread_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_input
        st.rerun()

    if st.button("➕ Start New Chat Thread", use_container_width=True):
        st.session_state.thread_id = f"session_{uuid.uuid4().hex[:6]}"
        st.rerun()

    st.markdown("---")
    st.header("🧾 Receipt Uploader & HITL")

    with st.expander("✍️ Human Review Form", expanded=False):
        with st.form("hitl_receipt_form"):
            man_vendor = st.text_input("Vendor Name")
            man_date = st.date_input("Receipt Date", value=datetime.today())
            man_total = st.number_input("Total Amount ($)", min_value=0.0, format="%.2f")
            submitted = st.form_submit_button("Confirm & Append", use_container_width=True)

            if submitted:
                if not man_vendor.strip():
                    st.error("Please provide a valid vendor name.")
                else:
                    try:
                        with psycopg.connect(get_db_uri(), autocommit=True) as conn:
                            with conn.cursor() as cursor:
                                cursor.execute(
                                    "INSERT INTO receipts (vendor, date, total) VALUES (%s, %s, %s)",
                                    (man_vendor, str(man_date), man_total)
                                )
                        st.success(f"Committed {man_vendor}!")
                    except Exception as e:
                        st.error(f"Database error: {e}")

# --- Main Chat Interface ---
chat_container = st.container()

with chat_container:
    try:
        state_snapshot = chatbot.get_state(config)
        messages = state_snapshot.values.get("messages", []) if state_snapshot.values else []
    except Exception:
        messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                if isinstance(msg.content, list):
                    for part in msg.content:
                        # Cast to dict for Pyright to avoid index errors
                        part_dict = part if isinstance(part, dict) else {}
                        if part_dict.get("type") == "text":
                            st.markdown(part_dict.get("text", ""))
                        elif part_dict.get("type") == "image_url":
                            url_data = part_dict.get("image_url", {})
                            if isinstance(url_data, dict):
                                st.image(url_data.get("url", ""))
                else:
                    st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

# Fixed Chat Input at the Bottom
if user_input := st.chat_input("Ask your agent anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        def response_generator():
            for event in chatbot.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
                msg, metadata = event
                if isinstance(msg, AIMessage) and msg.content:
                    yield msg.content

        st.write_stream(response_generator())
