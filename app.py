"""
app.py
------
Streamlit chat UI for the Zillow RAG Agent.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from data_ingestion import download_datasets, build_documents
from vector_store import build_vectorstore, load_vectorstore, vectorstore_exists
from rag_agent import build_agent, ask

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zillow RAG Agent",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Zillow Housing Market RAG Agent")
st.caption("Powered by LangChain · Ollama (Llama 3.1) · ChromaDB · 100% local")

# ---------------------------------------------------------------------------
# Sidebar — setup controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    if st.button("1️⃣  Download Zillow Data", use_container_width=True):
        with st.spinner("Downloading from Zillow Research..."):
            download_datasets()
        st.success("Data downloaded!")

    if st.button("2️⃣  Build Vector Store", use_container_width=True):
        with st.spinner("Embedding documents (this takes 2–5 min first time)..."):
            docs = build_documents()
            build_vectorstore(docs)
        st.success(f"Embedded {len(docs)} documents!")
        st.cache_resource.clear()   # ← clears all cached resources
        st.rerun()  

    st.divider()
    st.markdown("**Sample questions to try:**")
    sample_qs = [
        "Which metros had the highest home value growth in the last 12 months?",
        "What is the current median rent in Austin, TX?",
        "Compare home values between Seattle and Denver.",
        "Which markets are cooling down based on inventory trends?",
        "What is the rental trend in Nashville over the past year?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

    st.divider()
    st.info("**Privacy:** All processing is local. No data sent externally.")

# ---------------------------------------------------------------------------
# Load agent (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading RAG agent...")
def get_agent():
    if not vectorstore_exists():
        return None
    vs = load_vectorstore()
    return build_agent(vs)

agent = get_agent()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📂 Source documents used"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['metro']}** | {s['dataset']} | "
                        f"Latest: ${s['latest_value']:,} | "
                        f"12m change: {s['pct_change']}%"
                    )

# Handle prefilled question from sidebar
prefill = st.session_state.pop("prefill", None)

# Chat input
question = st.chat_input("Ask about US housing market trends...") or prefill

if question:
    if agent is None:
        st.error("⚠️ Vector store not found. Use the sidebar to Download Data and Build Vector Store first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching Zillow data..."):
                response = ask(agent, question)

            st.markdown(response["answer"])

            if response["sources"]:
                with st.expander("📂 Source documents used"):
                    for s in response["sources"]:
                        st.markdown(
                            f"- **{s['metro']}** | {s['dataset']} | "
                            f"Latest: ${s['latest_value']:,} | "
                            f"12m change: {s['pct_change']}%"
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
