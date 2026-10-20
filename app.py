import os
import re
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("RAG_DB_PATH", os.path.join(BASE_DIR, "chroma_db"))
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")

if not os.path.exists(DB_PATH):
    print("❌ Database non trovato. Esegui prima: python ingest.py")
    exit()

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Recuperiamo più candidati e li filtriamo prima di passarli a Llama.
CANDIDATE_K = 12
MAX_CONTEXT_DOCS = 6

dati_db = vectorstore.get()
tutti_i_documenti = [
    Document(page_content=testo, metadata=meta)
    for testo, meta in zip(dati_db["documents"], dati_db["metadatas"])
]


def nome_file(doc):
    source = doc.metadata.get("source", "Documento")
    return re.split(r"[\\/]", source)[-1]


def normalizza_nome_file(nome):
    """
    Normalizza il nome per riconoscere copie come:
      'ZTE MF297D VoIP.docx'
      'ZTE MF297D VoIP _1.docx'
    """
    base, ext = os.path.splitext(nome)
    base = re.sub(r"\s*_\d+$", "", base)
    base = re.sub(r"\s+(copy|copia|backup|bk|bkp|test|prova|draft|bozza|temp)$",
                  "", base, flags=re.IGNORECASE)
    return (base.strip().lower(), ext.lower())


def e_probabile_copia(nome):
    base, _ = os.path.splitext(nome)
    return bool(
        re.search(r"\s*_\d+$", base)
        or re.search(
            r"(?:^|[\s_-])(copy|copia|backup|bk|bkp|test|prova|draft|bozza|temp)(?:$|[\s_-])",
            base,
            flags=re.IGNORECASE,
        )
    )


def estrai_modelli(query):
    """
    Estrae token che sembrano modelli tecnici, es. MF297D, HH516V, NR7303.
    È volutamente conservativo: almeno una lettera e una cifra.
    """
    candidati = re.findall(r"\b[A-Za-z]{1,6}[-_]?[A-Za-z0-9]{2,}\b", query)
    esclusi = {
        "APN", "TR069", "TR-069", "CWMP", "USP", "IPV4", "IPV6",
        "SHALL", "SHOULD", "MUST", "MAY", "FWA", "CPE", "ODU", "IDU"
    }
    modelli = []
    for token in candidati:
        compatto = re.sub(r"[-_]", "", token).upper()
        if token.upper() in esclusi:
            continue
        if re.search(r"[A-Z]", compatto) and re.search(r"\d", compatto):
            modelli.append(token.upper())
    return list(dict.fromkeys(modelli))


def doc_corrisponde_modello(doc, modelli):
    if not modelli:
        return True
    haystack = (nome_file(doc) + " " + doc.page_content[:500]).upper()
    return any(m.upper() in haystack for m in modelli)


def token_query(query):
    stop = {
        "come", "posso", "sul", "sulla", "sullo", "sui", "sulle", "del", "della",
        "dello", "dei", "delle", "nel", "nella", "nello", "un", "una", "il", "lo",
        "la", "i", "gli", "le", "di", "a", "da", "per", "con", "che", "è", "e",
        "dispositivo", "device"
    }
    toks = re.findall(r"[A-Za-z0-9_.+-]+", query.lower())
    return [t for t in toks if len(t) > 1 and t not in stop]


def lexical_score(doc, query):
    """
    Piccolo score lessicale locale: non sostituisce BM25, ma migliora
    la V1 senza richiedere nuove librerie.
    """
    q = token_query(query)
    if not q:
        return 0.0
    text = (nome_file(doc) + " " + doc.page_content).lower()
    matches = sum(1 for t in q if t in text)
    return matches / len(q)


def recupera_documenti(query):
    modelli = estrai_modelli(query)

    # 1) Candidati semantici con score Chroma.
    # Chroma restituisce distance: più bassa = più vicino.
    candidati = vectorstore.similarity_search_with_score(query, k=CANDIDATE_K)

    ranked = []
    for doc, distance in candidati:
        # Esclude copie/backup/test/bozze PRIMA del retrieval finale e del contesto LLM.
        if e_probabile_copia(nome_file(doc)):
            continue

        # Se la domanda contiene un modello esplicito, scarta altri modelli.
        if not doc_corrisponde_modello(doc, modelli):
            continue

        lex = lexical_score(doc, query)

        # Convertiamo la distance in un peso monotono semplice solo per il ranking.
        semantic_weight = 1.0 / (1.0 + max(float(distance), 0.0))
        combined = 0.75 * semantic_weight + 0.25 * lex
        ranked.append((combined, doc))

    # 2) Aggiunge eventuali match lessicali forti dal DB, sempre rispettando il modello.
    for doc in tutti_i_documenti:
        # Anche i match lessicali devono rispettare l'esclusione delle copie.
        if e_probabile_copia(nome_file(doc)):
            continue
        if not doc_corrisponde_modello(doc, modelli):
            continue
        lex = lexical_score(doc, query)
        if lex >= 0.60:
            ranked.append((0.50 + 0.25 * lex, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # 3) Deduplica contenuto e copie di file.
    risultati = []
    testi_visti = set()
    file_canonici_visti = set()

    for score, doc in ranked:
        testo_key = re.sub(r"\s+", " ", doc.page_content.strip().lower())
        if testo_key in testi_visti:
            continue

        nome = nome_file(doc)
        canonico = normalizza_nome_file(nome)

        # Preferisce il file non marcato come copia.
        if e_probabile_copia(nome) and canonico in file_canonici_visti:
            continue

        testi_visti.add(testo_key)
        file_canonici_visti.add(canonico)
        risultati.append(doc)

        if len(risultati) >= MAX_CONTEXT_DOCS:
            break

    return risultati, modelli


llm = ChatOllama(model=LLM_MODEL, temperature=0)

template = """Sei un assistente tecnico.
Rispondi alla domanda usando ESCLUSIVAMENTE le informazioni presenti nel contesto fornito.

REGOLE:
1. Non usare conoscenza generale e non inventare informazioni.
2. Se la domanda cita un modello/dispositivo specifico, usa solo evidenze riferite a quel modello.
3. Riporta un Requirement ID e un livello normativo (SHALL/SHOULD/MUST/MAY)
   SOLO se sono esplicitamente presenti nella fonte. Non trasformare titoli,
   nomi di menu, parametri o frasi procedurali in Requirement ID o livelli normativi.
4. Se la domanda chiede "come" eseguire un'operazione, ricostruisci in ordine
   tutti i passaggi pertinenti presenti nel contesto.
5. Non includere informazioni non pertinenti alla domanda anche se presenti nel contesto.
6. Cita SEMPRE nel testo almeno una fonte nel formato esatto [Fonte N] per ogni passaggio o affermazione tecnica.
   Non ripetere il nome del file nella risposta: usa solo [Fonte N].
7. Se più passaggi consecutivi derivano dalla stessa fonte, puoi citarla alla fine del gruppo di passaggi.
8. Non aggiungere note, prerequisiti, limitazioni o informazioni collaterali se la fonte non li collega ESPLICITAMENTE all'operazione richiesta.
   In particolare, non collegare automaticamente sezioni vicine o argomenti diversi (es. TR069 e APN).
   Non trasformare un passaggio procedurale in una condizione tecnica più forte.
   Se la fonte indica che un'azione deve essere eseguita preliminarmente, riportala come passaggio della procedura.
   Esempio: "Prima di modificare l'APN è necessario disattivare i dati dalla homepage [Fonte N]."
9. Se il contesto non contiene informazioni sufficienti, dichiaralo chiaramente.

Contesto:
{context}

Domanda: {question}

Risposta:"""

prompt = ChatPromptTemplate.from_template(template)

print("\n--- Assistente RAG V2 Pronto! Scrivi una domanda o 'esci' ---")

while True:
    domanda = input("\nDomanda: ")
    if domanda.strip().lower() in ["esci", "exit", "quit"]:
        break
    if not domanda.strip():
        continue

    docs, modelli = recupera_documenti(domanda)

    if modelli:
        print(f"\n🔎 Modello/i rilevato/i: {', '.join(modelli)}")

    if not docs:
        print("\nNon sono state recuperate fonti sufficientemente pertinenti.")
        continue

    blocchi = []
    fonti = {}
    for i, doc in enumerate(docs, start=1):
        nf = nome_file(doc)
        fonti[i] = nf
        blocchi.append(f"[Fonte {i}]:\n{doc.page_content}\n")

    messaggi = prompt.format_messages(
        context="\n".join(blocchi),
        question=domanda
    )
    risposta = llm.invoke(messaggi)

    print(f"\n{risposta.content}")
    print("\n" + "=" * 50)
    print("📁 Documenti di riferimento:")

    # Deduplica solo la visualizzazione finale delle fonti.
    # Il contesto continua a contenere più chunk dello stesso documento.
    # Se esiste una fonte non-copia, le varianti _1/copy/backup vengono nascoste.
    gruppi = {}
    ordine = []

    for num, nf in fonti.items():
        canonico = normalizza_nome_file(nf)
        if canonico not in gruppi:
            gruppi[canonico] = []
            ordine.append(canonico)
        gruppi[canonico].append((num, nf))

    for canonico in ordine:
        candidati = gruppi[canonico]

        # Preferisce un nome non marcato come copia.
        non_copie = [(num, nf) for num, nf in candidati if not e_probabile_copia(nf)]
        scelto_num, scelto_nome = (non_copie[0] if non_copie else candidati[0])

        # Evita di mostrare più volte lo stesso file logico.
        print(f"  -> {scelto_nome}")

    print("=" * 50)
