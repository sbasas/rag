"""
vector_store.py
---------------
Embeds LangChain Documents into a local ChromaDB vector store
using Ollama's nomic-embed-text model (100% offline, no API calls).
"""

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
EMBED_MODEL     = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
COLLECTION_NAME = "zillow_research"


def get_embeddings() -> OllamaEmbeddings:
    """Return local Ollama embedding model (nomic-embed-text)."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_vectorstore(documents: list) -> Chroma:
    """
    Embed documents and persist to ChromaDB on disk.
    Safe to call multiple times — will overwrite existing store.
    """
    print(f"  [embed] embedding {len(documents)} documents via {EMBED_MODEL}...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"  [ok] vectorstore persisted to {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an already-built ChromaDB store from disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTORSTORE_DIR,
    )


def vectorstore_exists() -> bool:
    """Check whether a vectorstore has been built already."""
    chroma_db = os.path.join(VECTORSTORE_DIR, "chroma.sqlite3")
    return os.path.exists(chroma_db)
