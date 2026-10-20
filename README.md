# Local Technical Documentation RAG Assistant (Ollama + ChromaDB)

A 100% local **Retrieval-Augmented Generation (RAG)** system built in Python for querying technical documentation, requirement specifications, and CPE/FWA device manuals (`.docx`, `.pdf`, `.txt`), powered by **LangChain**, **ChromaDB**, and **Ollama**.

---

## Architecture

The workflow is divided into two independent stages:

### 1. Document Ingestion (`ingest.py`)
Reads source documents from the `Documenti/` folder, splits them into semantic chunks, generates numerical embeddings using the local **`nomic-embed-text`** model, and stores them persistently in `chroma_db/`:
- **Word Documents (`.docx`)**:
  - Regular paragraphs are split using `RecursiveCharacterTextSplitter` (`chunk_size=700`, `chunk_overlap=100`).
  - **Tables** are extracted row-by-row as dedicated documents (`[Tabella]`), ensuring technical requirements and tabular parameters are never truncated mid-row.
- **PDF (`.pdf`) and Text (`.txt`) Files**: loaded via `PyPDFLoader` and `TextLoader` and split into chunks.
- During ingestion, the generative LLM (`llama3.2`) is never loaded into RAM.

### 2. Hybrid Retrieval & Question Answering (`app.py`)
Accepts user queries from the command line and performs hybrid retrieval before generating grounded responses with **`llama3.2`** (`temperature=0`):
1. **Automatic Device Model Detection**: detects technical hardware model codes in the user query (e.g., `MF297D`, `NR7303`, `HH516V`) while distinguishing them from protocol acronyms (`TR-069`, `APN`, `CWMP`, `IPv4`, `FWA`, `CPE`), filtering context strictly to documents matching the target device.
2. **Duplicate & Draft Exclusion**: automatically filters out duplicate or temporary files (`_1`, `copy`, `copia`, `backup`, `bozza`, `draft`, `test`) prior to context assembly.
3. **Hybrid Ranking (Semantic + Lexical)**: combines ChromaDB vector similarity (`75%`) with a local lexical token-overlap score (`25%`), and supplements results with strong lexical matches (`>= 60%`).
4. **Strict Source Grounding & Citations**: instructs the LLM to answer exclusively from the retrieved context, citing every technical statement or procedural step using `[Fonte N]` and listing deduplicated reference files at the end of each answer.

---

## Project Structure

```text
RAG/
├── app.py              # Interactive CLI assistant (Hybrid retrieval + ChatOllama)
├── ingest.py           # Document loading, table-aware chunking, and ChromaDB indexing
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── Documenti/          # Input directory for .docx, .pdf, and .txt files to index
```

---

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** installed and running locally (`http://localhost:11434`).
3. Pull the required embedding and language models in Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
   Verify that Ollama is running and the models are available:
   ```bash
   ollama list
   curl http://localhost:11434
   ```

---

## Installation & Usage

### Linux / macOS
```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd RAG

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your documents (.docx, .pdf, .txt) inside Documenti/ and run ingestion
python ingest.py

# 5. Launch the interactive RAG assistant
python app.py
```

### Windows (PowerShell / CMD)
```bat
cd RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: Index documents located in Documenti\
python ingest.py

:: Start the interactive CLI assistant
python app.py
```

---

## Optional Environment Variables

Paths and model names default to relative project paths, but can be overridden via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RAG_DOCS_DIR` | `./Documenti` | Path to the directory containing source documents to index |
| `RAG_DB_PATH` | `./chroma_db` | Path to the persistent ChromaDB vector store |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama model used for generating embeddings |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Ollama model used for generating answers |
