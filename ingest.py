# ingest.py
# ----------------------------------------
import os
import sys
import time
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

# ----------------------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingest.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ----------------------------------------
# Folders to scan
FOLDERS = ["docs", "data"]
DB_FAISS_PATH = "vectorstore.faiss"

# ----------------------------------------
def load_all_documents():
    documents = []
    supported = [".txt", ".pdf", ".docx", ".md", ".csv", ".html"]

    for folder in FOLDERS:
        if not os.path.exists(folder):
            logging.warning(f"Folder not found: {folder}")
            continue

        logging.info(f"Scanning: {folder}/")

        for root, _, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                file_path = os.path.join(root, file)

                if ext not in supported:
                    logging.warning(f"Skipping unsupported file: {file_path}")
                    continue

                logging.info(f"Loading: {file_path}")

                try:
                    if ext == ".txt":
                        loader = TextLoader(file_path, encoding="utf-8")
                    elif ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                    elif ext == ".docx":
                        loader = Docx2txtLoader(file_path)
                    elif ext == ".md":
                        loader = UnstructuredMarkdownLoader(file_path)
                    elif ext == ".csv":
                        loader = CSVLoader(file_path)
                    elif ext == ".html":
                        loader = UnstructuredHTMLLoader(file_path)

                    docs = loader.load()
                    documents.extend(docs)

                except Exception as e:
                    logging.error(f"Failed to load {file_path}: {e}")

    return documents

# ----------------------------------------
def choose_embedding():
    mode = sys.argv[1] if len(sys.argv) > 1 else "huggingface"
    mode = mode.lower()

    if mode == "huggingface":
        model = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Using HuggingFace embeddings: {model}")
        return HuggingFaceEmbeddings(model_name=model)

    elif mode == "ollama":
        model = "nomic-embed-text"
        logging.info(f"Using Ollama embeddings: {model}")
        return OllamaEmbeddings(model=model)

    else:
        raise ValueError("Invalid mode. Use: huggingface OR ollama")

# ----------------------------------------
if __name__ == "__main__":
    start = time.time()
    logging.info("🚀 Starting ingestion...")

    # Load docs
    documents = load_all_documents()
    logging.info(f"Total loaded documents: {len(documents)}")

    if not documents:
        logging.error("No supported files found in docs/ or data/. Exiting.")
        sys.exit(1)

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = splitter.split_documents(documents)
    logging.info(f"Created {len(texts)} text chunks.")

    # Embeddings
    embeddings = choose_embedding()

    # Build vectorstore
    db = FAISS.from_documents(texts, embeddings)

    # Save vectorstore
    db.save_local(DB_FAISS_PATH)
    logging.info(f"💾 Saved FAISS vectorstore to {DB_FAISS_PATH}")

    total = time.time() - start
    logging.info(f"✅ Ingestion completed in {total:.2f} seconds!")
