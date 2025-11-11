# ingest.py
# -------------------------------
import os
import glob
import time
import sys
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# STEP 1: Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingest.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# -------------------------------
# STEP 2: Paths
DATA_PATH = "./data"                  # Folder containing your .txt files
DB_FAISS_PATH = "vectorstore.faiss"   # Name for FAISS vector database

# -------------------------------
# STEP 3: Load Documents
start_time = time.time()
logging.info("Starting ingestion process...")
logging.info(f"Scanning folder: {DATA_PATH}")

files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
documents = []

if not files:
    logging.warning("No .txt files found in ./data. Please add at least one file and rerun.")
    exit()

for file in files:
    logging.info(f"Loading file: {file}")
    try:
        loader = TextLoader(file, encoding="utf-8")
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        logging.info(f"Loaded {len(loaded_docs)} document(s) from {file}")
    except Exception as e:
        logging.error(f"Failed to load {file}: {e}")

logging.info(f"Total documents loaded: {len(documents)}")

# -------------------------------
# STEP 4: Split Text
logging.info("Splitting documents into smaller chunks...")
split_start = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
logging.info(f"Split into {len(texts)} chunks in {time.time() - split_start:.2f}s")

# -------------------------------
# STEP 5: Create Embeddings (Toggle between Ollama / Hugging Face)
logging.info("Generating embeddings...")

# Optional CLI toggle: python ingest.py huggingface OR python ingest.py ollama
EMBEDDING_MODE = sys.argv[1] if len(sys.argv) > 1 else "huggingface"
embed_start = time.time()

try:
    if EMBEDDING_MODE.lower() == "huggingface":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Using Hugging Face model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    elif EMBEDDING_MODE.lower() == "ollama":
        model_name = "nomic-embed-text"
        logging.info(f"Using Ollama model: {model_name}")
        embeddings = OllamaEmbeddings(model=model_name)

    else:
        raise ValueError("Invalid EMBEDDING_MODE. Use 'huggingface' or 'ollama'.")

    db = FAISS.from_documents(texts, embeddings)
    logging.info(f"Embeddings created successfully using {EMBEDDING_MODE} in {time.time() - embed_start:.2f}s")

except Exception as e:
    logging.error(f"Embedding generation failed: {e}")
    raise e

# -------------------------------
# STEP 6: Save FAISS
logging.info("Saving embeddings into FAISS vector database...")
save_start = time.time()

try:
    db.save_local(DB_FAISS_PATH)
    logging.info(f"Vector database saved as '{DB_FAISS_PATH}' in {time.time() - save_start:.2f}s")
except Exception as e:
    logging.error(f"Failed to save FAISS vector database: {e}")
    raise e

# -------------------------------
# STEP 7: Finish
total_time = time.time() - start_time
logging.info(f"Ingestion complete in {total_time:.2f} seconds!")
logging.info("Your FAISS vectorstore is ready to use.")
