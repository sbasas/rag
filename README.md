# 🏠 Zillow Housing Market RAG Agent

A fully local RAG (Retrieval-Augmented Generation) agent that answers natural language questions about US housing market trends using public Zillow Research data.

**100% private — no data leaves your machine.**

---

## Architecture

```
Zillow Research CSVs (public)
        ↓
  data_ingestion.py        ← downloads + converts to LangChain Documents
        ↓
  vector_store.py          ← embeds via nomic-embed-text (Ollama), stores in ChromaDB
        ↓
  rag_agent.py             ← retrieves + answers via Llama 3.1 8B (Ollama)
        ↓
  app.py                   ← Streamlit chat UI
```

## Tech Stack

| Component      | Tool                          |
|---------------|-------------------------------|
| LLM           | Llama 3.1 8B via Ollama       |
| Embeddings    | nomic-embed-text via Ollama   |
| Vector Store  | ChromaDB (local disk)         |
| Orchestration | LangChain                     |
| UI            | Streamlit                     |
| Data          | Zillow Research (public CSV)  |

---

## Setup

### 1. Prerequisites

Install [Ollama](https://ollama.com) then pull the required models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. First-time setup (in the sidebar)

1. Click **Download Zillow Data** — fetches 4 datasets from Zillow Research
2. Click **Build Vector Store** — embeds ~200 documents (takes 2–5 min once)
3. Start chatting!

---

## Datasets Included

| Dataset               | Description                          |
|----------------------|--------------------------------------|
| `median_sale_price`  | Zillow Home Value Index by metro     |
| `rental_index`       | Zillow Observed Rent Index (ZORI)    |
| `days_to_pending`    | Market Temperature Index             |
| `inventory`          | For-Sale Inventory by metro          |

---

## Sample Questions

- *"Which metros had the highest home value growth in the last 12 months?"*
- *"What is the current median rent in Austin, TX?"*
- *"Compare home values between Seattle and Denver."*
- *"Which markets are cooling down based on inventory trends?"*
- *"What is the rental trend in Nashville over the past year?"*

---

## Project Structure

```
zillow-rag-agent/
├── app.py                  # Streamlit UI
├── requirements.txt
├── src/
│   ├── data_ingestion.py   # Download + parse Zillow CSVs
│   ├── vector_store.py     # ChromaDB embedding + retrieval
│   └── rag_agent.py        # LangChain RAG chain
├── data/                   # Downloaded CSVs (auto-created)
└── vectorstore/            # ChromaDB files (auto-created)
```

---

## Privacy

All computation happens locally:
- Ollama runs LLMs on your machine
- ChromaDB stores vectors on disk
- No API keys required
- No network calls after initial data download
