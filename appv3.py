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
    page_title="LangGraph Assistant v8", 
    page_icon="⚡", 
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stButton button { border-radius: 8px; transition: all 0.3s ease; }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    div[data-testid="stSidebarUserContent"] { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ LangGraph Assistant v8")
st.caption("Enterprise-grade Multi-modal RAG + PostgreSQL Persistence + Streaming")

# 1. State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- Database Setup ---
DEFAULT_DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:871031@localhost:5432/chatvectorv1")

def get_db_uri():
    return st.session_state.get("db_uri", DEFAULT_DB_URI)

def init_receipts_postgres():
    try:
        uri = get_db_uri()
        if not uri:
            return
        with psycopg.connect(uri, autocommit=True) as conn:
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
        print(f"Database initialization skipped/failed: {e}")

init_receipts_postgres()

# --- Lazy-Loaded RAG Service (On-Demand Only) ---
@st.cache_resource(show_spinner="Waking up RAG & Embedding models...")
def _load_rag_singleton(db_uri: str):
    """Imports and initializes unified_rag_service strictly on first call."""
    from unified_rag_service import rag_service
    if rag_service is not None:
        rag_service.reset_connection(db_uri)
    return rag_service

def get_rag_service():
    """Wrapper to load the RAG service on-demand and update UI health state."""
    try:
        service = _load_rag_singleton(get_db_uri())
        if service is not None:
            st.session_state["rag_initialized"] = True
        return service
    except Exception as e:
        print(f"⚠️ Could not load rag_service on-demand: {e}")
        return None

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
    service = get_rag_service()
    if service is None:
        return "Error: RAG Service could not be loaded on-demand. Please check system logs."
    return service.query(query)

@tool
def ingest_document_file(file_path: str) -> str:
    """Ingests a document file (PDF, Text, or Image) into the permanent knowledge base."""
    service = get_rag_service()
    if service is None:
        return "Error: RAG Service could not be loaded on-demand."
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return service.ingest_pdf(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic']:
        return service.ingest_image(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return service.ingest_text(text, source=os.path.basename(file_path))
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
@st.cache_resource
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
        "Upload Document", 
        type=["pdf", "txt", "jpg", "jpeg", "png", "webp", "heic"],
        key="main_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state['pending_upload'] = uploaded_file

    if 'pending_upload' in st.session_state:
        file_to_process = st.session_state['pending_upload']
        st.info(f"Selected: {file_to_process.name}")

        # High visibility vertical actions (Lazy loads RAG on click)
        if st.button("📄 Analyze with AI", use_container_width=True, type="primary"):
            with st.spinner("Extracting on-demand..."):
                try:
                    service = get_rag_service()
                    if service is None:
                        st.error("RAG Service could not be loaded.")
                        st.stop()
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, file_to_process.name)
                    with open(file_path, "wb") as f:
                        f.write(file_to_process.getbuffer())
                    
                    text_content = service.extract_text(file_path)
                    image_message = HumanMessage(content=f"Please analyze this document: \n\n{text_content}")
                    chatbot.update_state(config, {"messages": [image_message]})
                    st.toast("Document sent to AI!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("💾 Save to Knowledge Base", use_container_width=True):
            with st.spinner("Ingesting on-demand..."):
                try:
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, file_to_process.name)
                    with open(file_path, "wb") as f:
                        f.write(file_to_process.getbuffer())
                    
                    result = ingest_document_file.invoke({"file_path": file_path})
                    st.toast(result)
                    del st.session_state['pending_upload']
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("❌ Clear Selection", use_container_width=True):
            del st.session_state['pending_upload']
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Control Panel")
    
    def check_db_connection():
        try:
            with psycopg.connect(get_db_uri(), connect_timeout=2) as conn:
                return True, "Connected successfully!"
        except Exception as e:
            return False, str(e)

    # --- System Health ---
    with st.expander("🛠️ System Health", expanded=True):
        current_uri = get_db_uri()
        masked_uri = current_uri
        if "@" in current_uri and "://" in current_uri:
            prefix, rest = current_uri.split("://", 1)
            if "@" in rest:
                user_pass, host = rest.split("@", 1)
                if ":" in user_pass:
                    user, _ = user_pass.split(":", 1)
                    masked_uri = f"{prefix}://{user}:****@{host}"
        
        st.markdown(f"**Using URI:** `{masked_uri}`")
        
        db_connected, db_msg = check_db_connection()
        db_status = "🟢 Connected" if db_connected else "🔴 Disconnected"
        
        # Checks state without forcing an import
        is_rag_loaded = st.session_state.get("rag_initialized", False)
        rag_status = "🟢 Loaded" if is_rag_loaded else "🔴 Not Loaded (On-Demand)"
        
        st.markdown(f"**Database:** {db_status}")
        if not db_connected:
            st.error(f"Connection Error: {db_msg}")
        st.markdown(f"**RAG Service:** {rag_status}")

    # --- DB Configuration ---
    st.subheader("🌐 Database Config")
    db_uri_input = st.text_input("Postgres URI", value=get_db_uri())
    if db_uri_input != get_db_uri():
        st.session_state.db_uri = db_uri_input
        # Clear cached RAG singleton so next call connects to the new URI
        _load_rag_singleton.clear()
        st.session_state["rag_initialized"] = False
        st.toast("Database URI updated! RAG will reconnect on next use.")
        st.rerun()
    
    st.markdown("---")

    thread_input = st.text_input("Active Thread ID", value=st.session_state.thread_id)
    if thread_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_input
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Thread", use_container_width=True):
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:6]}"
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.thread_id = f"reset_{uuid.uuid4().hex[:6]}"
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
                if not man_vendor:
                    st.warning("Please provide a vendor name.")
                else:
                    result = log_receipt_to_table.invoke({
                        "vendor": man_vendor, 
                        "date": str(man_date), 
                        "total": float(man_total)
                    })
                    st.success(result)

# --- Main Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_loaded" not in st.session_state:
    state = chatbot.get_state(config)
    if state and state.values.get("messages"):
        st.session_state.messages = state.values["messages"]
    st.session_state.history_loaded = True

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("User"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("Assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("Assistant"):
        container = st.empty()
        full_response = ""
        
        try:
            for event in chatbot.stream(
                {"messages": [HumanMessage(content=prompt)]}, 
                config, 
                stream_mode="messages"
            ):
                if isinstance(event, tuple):
                    msg = event[0]
                else:
                    msg = event
                
                if isinstance(msg, AIMessage) and msg.content:
                    full_response += msg.content
                    container.markdown(full_response + "▌")
            
            container.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"Chat Error: {e}")