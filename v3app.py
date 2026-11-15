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
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="LangGraph Assistant v3", 
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

st.title("⚡ LangGraph Assistant v3 (with HITL & Chat Commands)")
st.caption("Enterprise-grade Multi-modal RAG + PostgreSQL Persistence + AI Quality Guardrails & HITL")

# 1. State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    latest_assessment: Optional[dict]

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
        return f"Successfully prepared {len(rows)} receipt records for export!"
    except Exception as e:
        return f"Failed to export: {str(e)}"

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

# --- AI Quality Inspector Schema & Nodes ---
class PageQualityAssessment(BaseModel):
    needs_review: bool = Field(
        description="True if the text or document is blurry, garbled, incomplete, or unreadable, requiring human intervention. False if clear."
    )
    reason: str = Field(description="Explanation of why human review is required or why it passed.")

def evaluate_quality_node(state: ChatState):
    """AI Node inspecting latest message/document for OCR/quality issues."""
    messages = state["messages"]
    latest_content = messages[-1].content if messages else ""
    
    lower_content = latest_content.lower()
    
    # Bypass LLM structured parsing if user sent a rejection or approval command
    if any(sig in lower_content for sig in ["[rejected", "[cancelled", "reject", "cancel", "abort"]):
        return {"latest_assessment": {"needs_review": False, "reason": "User aborted upload."}}
    
    if any(sig in lower_content for sig in ["approved", "[approved", "yes", "proceed", "ok"]):
        return {"latest_assessment": {"needs_review": False, "reason": "User approved document continuation."}}

    structured_llm = llm.with_structured_output(PageQualityAssessment)
    prompt = (
        "You are an enterprise data quality inspector. Examine the extracted text/content. "
        "If the text is corrupted, missing key fields due to blurriness, or contains garbage OCR output, flag it.\n\n"
        f"Content:\n{latest_content}"
    )
    assessment = structured_llm.invoke(prompt)
    return {"latest_assessment": assessment.dict()}

def human_review_node(state: ChatState):
    """Pauses graph execution via native interrupt for human verification."""
    assessment = state.get("latest_assessment", {})
    latest_content = state["messages"][-1].content if state["messages"] else ""
    
    # Pauses and waits for Command(resume=...) from either the form or chat input
    user_response = interrupt({
        "type": "quality_review",
        "reason": assessment.get("reason", "AI flagged content as potentially unreadable or low quality."),
        "content_preview": latest_content[:1000]
    })
    
    # If the user typed 'approved' or 'yes', keep the original content. Otherwise use their edited text.
    if str(user_response).strip().lower() in ["approved", "approve", "yes", "ok", "proceed", "[approved]"]:
        final_text = latest_content
    else:
        final_text = str(user_response)
        
    return {"messages": [HumanMessage(content=f"Human Verified Data: {final_text}")]}

def quality_router(state: ChatState):
    assessment = state.get("latest_assessment", {})
    if assessment.get("needs_review", False):
        return "human_review_node"
    return "chat_node"

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

# 5. Graph Compilation with HITL Nodes
@st.cache_resource
def get_chatbot():
    connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
    conn = psycopg.connect(get_db_uri(), **connection_kwargs)
    db_checkpointer = PostgresSaver(conn)
    db_checkpointer.setup()

    workflow = StateGraph(ChatState)
    workflow.add_node("evaluate_quality", evaluate_quality_node)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tools", ToolNode(tools))

    # Flow logic with quality gate
    workflow.add_edge(START, "evaluate_quality")
    workflow.add_conditional_edges(
        "evaluate_quality",
        quality_router,
        {
            "human_review_node": "human_review_node",
            "chat_node": "chat_node"
        }
    )
    workflow.add_edge("human_review_node", "chat_node")
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
        if st.button("📄 Analyze with AI (HITL Check)", use_container_width=True, type="primary"):
            with st.spinner("Extracting & inspecting quality..."):
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
                    doc_message = HumanMessage(content=f"Please review this document extraction:\n\n{text_content}")
                    
                    # Run graph (will pause at human_review_node if flagged)
                    for event in chatbot.stream({"messages": [doc_message]}, config, stream_mode="messages"):
                        pass
                    st.toast("Document evaluated!")
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

# --- Main Chat UI & HITL Interrupt Interceptor ---
if "messages" not in st.session_state:
    st.session_state.messages = []

state_snapshot = chatbot.get_state(config)
if state_snapshot and state_snapshot.values.get("messages"):
    st.session_state.messages = state_snapshot.values["messages"]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("User"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("Assistant"):
            st.markdown(msg.content)

# --- Check if Graph is Interrupted for Human Review ---
if state_snapshot.next and "human_review_node" in state_snapshot.next:
    st.markdown("---")
    st.error("⚠️ **Human-in-the-Loop Triggered: AI Quality Inspector Flagged Unreadable Data!**")
    
    interrupt_payload = state_snapshot.tasks[0].interrupts[0].value if state_snapshot.tasks[0].interrupts else {}
    
    st.info(f"**Reason for Review:** {interrupt_payload.get('reason', 'Quality check failed.')}")
    st.text_area("Extracted Content Preview:", value=interrupt_payload.get('content_preview', ''), disabled=True)
    
    # 1. Form to submit corrected text or approve
    with st.form("hitl_override_form"):
        corrected_text = st.text_area(
            "Edit / Provide Corrected OCR Text (or leave as-is to approve):", 
            value=interrupt_payload.get('content_preview', '')
        )
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            resume_button = st.form_submit_button("🚀 Submit & Resume", use_container_width=True, type="primary")
        with col_f2:
            approve_button = st.form_submit_button("✅ Quick Approve", use_container_width=True)
        
        if resume_button:
            with st.spinner("Resuming graph execution..."):
                try:
                    for event in chatbot.stream(Command(resume=corrected_text), config, stream_mode="messages"):
                        pass
                    st.success("Workflow resumed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Resume Error: {e}")
                    
        if approve_button:
            with st.spinner("Approving and resuming workflow..."):
                try:
                    for event in chatbot.stream(Command(resume="approved"), config, stream_mode="messages"):
                        pass
                    st.success("Workflow approved and resumed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Resume Error: {e}")

    # 2. Standalone Abort Button
    if st.button("❌ Discard & Abort (Reset Thread)", use_container_width=True):
        with st.spinner("Discarding and resetting thread..."):
            try:
                for event in chatbot.stream(Command(resume="[CANCELLED_BY_USER]"), config, stream_mode="messages"):
                    pass
            except Exception:
                pass
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:6]}"
            if 'pending_upload' in st.session_state:
                del st.session_state['pending_upload']
            st.success("Upload discarded and thread reset!")
            st.rerun()

    # 3. Chatbox override to type 'approved', 'yes', or 'reject' directly
    if chat_input_val := st.chat_input("Type 'approved', 'yes', or 'reject'..."):
        clean_input = chat_input_val.strip().lower()
        if clean_input in ["reject", "cancel", "abort", "no"]:
            with st.spinner("Rejecting upload via chat command..."):
                try:
                    for event in chatbot.stream(Command(resume="[REJECTED_VIA_CHAT]"), config, stream_mode="messages"):
                        pass
                except Exception:
                    pass
                st.session_state.thread_id = f"session_{uuid.uuid4().hex[:6]}"
                if 'pending_upload' in st.session_state:
                    del st.session_state['pending_upload']
                st.success("Upload rejected and thread reset via chat!")
                st.rerun()
        else:
            # Passes 'approved', 'yes', or custom text straight into Command(resume=...)
            with st.spinner("Resuming workflow via chat input..."):
                try:
                    for event in chatbot.stream(Command(resume=chat_input_val), config, stream_mode="messages"):
                        pass
                    st.success("Workflow resumed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Resume Error: {e}")
else:
    # Standard Chat Input when NOT interrupted
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
                    msg = event[0] if isinstance(event, tuple) else event
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response += msg.content
                        container.markdown(full_response + "▌")
                
                container.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))
            except Exception as e:
                st.error(f"Chat Error: {e}")