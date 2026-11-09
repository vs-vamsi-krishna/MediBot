"""MedRAG — clean light-mode Streamlit UI.
Keeps the existing data preparation, ChromaDB, and Gemini backend unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

from config import BASE_DIR, CHROMA_DIR, GEMINI_MODEL, TOP_K
from data_processing import load_or_prepare_records
from rag_backend import (
    answer_question,
    get_embeddings,
    get_gemini_client,
    get_vectorstore,
)

st.set_page_config(
    page_title="MedRAG | FDA Drug Information",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = BASE_DIR / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def get_api_key() -> str | None:
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


load_css()


def download_chroma_from_huggingface():
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        return

    hf_token = st.secrets.get("HF_TOKEN", "") or os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing.")

    snapshot_download(
        repo_id="Vamsikrishna-05/medrag-chroma-db",
        repo_type="dataset",
        local_dir=str(CHROMA_DIR),
        token=hf_token,
    )


@st.cache_data(show_spinner="Loading prepared FDA records...")
def cached_records():
    return load_or_prepare_records()


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def cached_vectorstore(_embeddings):
    return get_vectorstore(_embeddings)


@st.cache_resource
def cached_gemini(api_key: str | None):
    return get_gemini_client(api_key)


with st.spinner("Preparing MedRAG..."):
    try:
        download_chroma_from_huggingface()
        records = cached_records()
        embeddings = cached_embeddings()
        vectorstore = cached_vectorstore(embeddings)
        startup_error = None
    except Exception as exc:
        records = None
        vectorstore = None
        startup_error = str(exc)

    api_key = get_api_key()
    gemini_client = cached_gemini(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

QUICK_QUESTIONS = [
    "What is paracetamol used for?",
    "Show details for application 000004",
    "What is the active ingredient in PAREDRINE?",
    "Show the latest submission for application 007337",
]

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✚</div>
            <div>
                <div class="brand-name">MedRAG</div>
                <div class="brand-subtitle">FDA Drug Information</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='side-label'>WORKSPACE</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item nav-active'>▣ &nbsp; Chat</div>", unsafe_allow_html=True)
    if st.button("▤  FDA Data", key="nav_fda", width="stretch"):
        st.session_state.show_data_info = not st.session_state.get("show_data_info", False)
    if st.button("✦  Sample Questions", key="nav_samples", width="stretch"):
        st.session_state.show_samples = not st.session_state.get("show_samples", False)
    if st.button("ⓘ  About MedRAG", key="nav_about", width="stretch"):
        st.session_state.show_about = not st.session_state.get("show_about", False)

    st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="trust-card">
            <div class="trust-icon">✓</div>
            <div class="trust-title">FDA data workspace</div>
            <div class="trust-copy">Answers about FDA records use the available local dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("＋  New conversation", key="new_chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    if st.button("Clear conversation", key="clear_chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
        
# Header
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown(
        """
        <div class="eyebrow"><span class="status-dot"></span> YOUR FDA DRUG INFORMATION ASSISTANT</div>
        <h1 class="page-title">Clear answers. Reliable records.</h1>
        <p class="page-subtitle">Ask about FDA products, applications, sponsors, strengths, dosage forms, marketing status, and submission history.</p>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        "<div class='powered-label'>Powered by<br><strong>Gemini</strong> ✦</div>",
        unsafe_allow_html=True,
    )

if startup_error:
    st.error("MedRAG could not load the FDA data or vector index. Check the setup details below.")
    with st.expander("Setup details"):
        st.code(startup_error)

# Compact feature cards
feature_cols = st.columns(4)
features = [
    ("⌕", "Drug information", "Product details and ingredients"),
    ("▤", "Applications", "Application type and sponsor"),
    ("◷", "Submissions", "Submission dates and status"),
    ("＋", "Medical Q&A", "General medicine information"),
]
for col, (symbol, title, desc) in zip(feature_cols, features):
    with col:
        st.markdown(
            f"<div class='feature-card'><div class='feature-symbol'>{symbol}</div><div class='feature-title'>{title}</div><div class='feature-desc'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

if st.session_state.get("show_data_info"):
    st.info("FDA answers are limited to information represented in your prepared FDA dataset and retrieved records.")
if st.session_state.get("show_samples"):
    st.markdown("**Sample questions**")
    for question in QUICK_QUESTIONS:
        if st.button(question, key=f"sample_{question}", width="stretch"):
            st.session_state.pending_question = question
            st.rerun()
if st.session_state.get("show_about"):
    st.info("MedRAG combines FDA-record retrieval with Gemini-generated responses. General medical answers are not a substitute for advice from a qualified healthcare professional.")

# Suggested prompts appear until a conversation begins.
if not st.session_state.messages:
    st.markdown("<div class='section-heading'>Try a question</div>", unsafe_allow_html=True)
    quick_cols = st.columns(2)
    for i, question in enumerate(QUICK_QUESTIONS):
        with quick_cols[i % 2]:
            if st.button(question, key=f"quick_{i}", width="stretch"):
                st.session_state.pending_question = question
                st.rerun()
else:
    st.markdown("<div class='section-heading'>Conversation</div>", unsafe_allow_html=True)
    
# Conversation history and source expander
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="💬" if message["role"] == "user" else "🩺"):
        st.markdown(message["content"])
        sources = message.get("sources")
        if message["role"] == "assistant" and sources is not None and not sources.empty:
            with st.expander(f"FDA records used · {len(sources)}"):
                st.dataframe(sources, hide_index=True, width="stretch")

pending = st.session_state.pop("pending_question", None)
user_query = st.chat_input("Ask about a drug, application number, sponsor, or medicine...")
query = pending or user_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="💬"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Finding the most relevant information..."):
            if records is None:
                result = {
                    "answer": "FDA search is not ready. Please check the setup details above.",
                    "sources": None,
                    "route": "fda",
                }
            else:
                result = answer_question(query, records, vectorstore, gemini_client)
        st.markdown(result.get("answer", "I couldn't generate an answer right now."))
        sources = result.get("sources")
        if sources is not None and not sources.empty:
            with st.expander(f"FDA records used · {len(sources)}"):
                st.dataframe(sources, hide_index=True, width="stretch")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", "I couldn't generate an answer right now."),
            "sources": result.get("sources"),
            "route": result.get("route"),
        }
    )
    st.rerun()

st.markdown(
    "<div class='footer-note'>MedRAG provides information from available FDA records and general AI-generated medical information. It is not a substitute for professional medical advice.</div>",
    unsafe_allow_html=True,
)
