"""
rag_agent.py
------------
LangChain RAG agent that answers questions about Zillow housing data
using a local Ollama LLM (Llama 3.1 8B) and ChromaDB retriever.

100% offline — no data leaves your machine.
"""

from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

LLM_MODEL    = "llama3.1"   # pull with: ollama pull llama3.1
TEMPERATURE  = 0.1           # low temp = factual, consistent answers
TOP_K_DOCS   = 5             # number of context chunks retrieved per query

# ---------------------------------------------------------------------------
# Prompt — instructs the LLM to stay grounded in retrieved Zillow data
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a real estate market analyst specializing in US housing trends.
You answer questions strictly using the Zillow research data provided below.

Guidelines:
- Always mention specific metro names and dates from the data
- Quote actual numbers (prices, rents, % changes) when available
- If the data does not contain enough information to answer, say so clearly
- Do not speculate beyond what the data shows
- Format dollar amounts with commas (e.g., $425,000)
- Keep answers concise but data-rich

Zillow Data Context:
{context}

Question: {question}

Answer:"""
)


def build_agent(vectorstore: Chroma) -> RetrievalQA:
    """
    Wire together the LLM + retriever into a RetrievalQA chain.

    Args:
        vectorstore: Loaded ChromaDB instance.

    Returns:
        A LangChain RetrievalQA chain ready for .invoke()
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",           # Maximum Marginal Relevance — diverse results
        search_kwargs={"k": TOP_K_DOCS},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # stuffs all retrieved docs into one prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
        return_source_documents=True,
    )

    return chain


def ask(chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the RAG chain.

    Returns:
        {
          "answer": str,
          "sources": [{"metro": ..., "dataset": ..., "latest_value": ...}]
        }
    """
    result = chain.invoke({"query": question})

    sources = [
        {
            "metro":   doc.metadata.get("metro", "N/A"),
            "dataset": doc.metadata.get("dataset", "N/A"),
            "latest_value": doc.metadata.get("latest_value", "N/A"),
            "pct_change":   doc.metadata.get("pct_change_12m", "N/A"),
        }
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer":  result["result"],
        "sources": sources,
    }
