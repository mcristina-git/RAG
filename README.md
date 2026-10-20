# Assistente RAG Locale per Documentazione Tecnica (Ollama + ChromaDB)

Sistema **Retrieval-Augmented Generation (RAG)** 100% locale in Python per l'interrogazione di documentazione tecnica, specifiche di requisito e manuali dispositivi CPE/FWA (`.docx`, `.pdf`, `.txt`), basato su **LangChain**, **ChromaDB** e **Ollama**.

---

## Architettura del Progetto

Il sistema è diviso in due fasi indipendenti:

### 1. Indicizzazione Documenti (`ingest.py`)
Legge i documenti presenti nella cartella `Documenti/`, li suddivide in frammenti (chunk) e genera i vettori numerici tramite il modello di embedding locale **`nomic-embed-text`** salvandoli in `chroma_db/`:
- **File Word (`.docx`)**:
  - I paragrafi di testo vengono divisi con `RecursiveCharacterTextSplitter` (`chunk_size=700`, `chunk_overlap=100`).
  - Le **tabelle** vengono estratte riga per riga come documenti dedicati (`[Tabella]`), evitando che requisiti tecnici o parametri tabellari vengano spezzati a metà.
- **File PDF (`.pdf`) e Testo (`.txt`)**: caricati tramite `PyPDFLoader` e `TextLoader` e suddivisi in chunk.
- In questa fase il modello linguistico (`llama3.2`) non viene caricato in RAM.

### 2. Interrogazione e Risposta (`app.py`)
Riceve le domande dell'utente da terminale ed esegue un recupero ibrido prima di generare la risposta con **`llama3.2`** (`temperature=0`):
1. **Riconoscimento automatico modello dispositivo**: identifica nella domanda sigle tecniche di modelli (es. `MF297D`, `NR7303`, `HH516V`) distinguendole da acronimi di protocollo (`TR-069`, `APN`, `CWMP`, `IPv4`, `FWA`, `CPE`), filtrando i soli documenti pertinenti al modello richiesto.
2. **Esclusione copie e bozze**: scarta automaticamente file duplicati o temporanei (`_1`, `copy`, `copia`, `backup`, `bozza`, `draft`, `test`).
3. **Ranking ibrido (Semantico + Lessicale)**: combina la distanza vettoriale di ChromaDB (`75%`) con uno score lessicale locale sui token della domanda (`25%`) e recupera anche eventuali match lessicali forti (`>= 60%`).
4. **Citazione rigorosa delle fonti**: istruisce l'LLM a rispondere esclusivamente sulla base del contesto recuperato, citando per ogni affermazione o passaggio procedurale la fonte nel formato `[Fonte N]` ed elencando a fine risposta i documenti di riferimento deduplicati.

---

## Struttura della Repository

```text
RAG/
├── app.py              # Assistente CLI interattivo (Retrieval ibrido + ChatOllama)
├── ingest.py           # Pipeline di caricamento, chunking e indicizzazione su ChromaDB
├── requirements.txt    # Dipendenze Python del progetto
├── README.md           # Documentazione completa del progetto
└── Documenti/          # Cartella in cui inserire i file .docx, .pdf, .txt da indicizzare
```

---

## Prerequisiti

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** installato e in esecuzione (`http://localhost:11434`).
3. Scaricare i due modelli richiesti in Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
   Per verificare che Ollama sia attivo e i modelli siano presenti:
   ```bash
   ollama list
   curl http://localhost:11434
   ```

---

## Installazione e Avvio

### Su Linux / macOS
```bash
# 1. Clona il repository ed entra nella cartella
git clone <URL_DEL_TUO_REPO_GITHUB>
cd RAG

# 2. Crea e attiva l'ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Inserisci i file (.docx, .pdf, .txt) nella cartella Documenti/ e avvia l'indicizzazione
python ingest.py

# 5. Avvia l'assistente RAG
python app.py
```

### Su Windows (PowerShell / CMD)
```bat
cd RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: Indicizza i documenti presenti in Documenti\
python ingest.py

:: Avvia l'assistente interattivo
python app.py
```

---

## Variabili d'Ambiente Opzionali

I percorsi e i modelli sono configurati automaticamente in modo relativo alla cartella del progetto, ma possono essere personalizzati tramite variabili d'ambiente:

| Variabile | Default | Descrizione |
| :--- | :--- | :--- |
| `RAG_DOCS_DIR` | `./Documenti` | Percorso della cartella contenente i documenti da indicizzare |
| `RAG_DB_PATH` | `./chroma_db` | Percorso del database vettoriale ChromaDB |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Modello Ollama usato per gli embedding |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Modello Ollama usato per generare le risposte |
