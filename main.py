import os
import warnings
import torch
import traceback
import subprocess
from flask import Flask, request, jsonify
from datetime import datetime
import csv
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate

# ---------------------------------
# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_FAISS_PATH = "vectorstore.faiss"

# ---------------------------------
# Run ingest.py automatically if FAISS database is missing
if not os.path.exists(DB_FAISS_PATH):
    logging.info("FAISS database not found. Running ingest.py...")
    try:
        subprocess.run([os.environ.get("PYTHON", "python"), "ingest.py"], check=True)
        logging.info("ingest.py completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ingest.py failed: {e}")
        # Don't crash here; depending on your preference you can exit(1) instead.
        # exit(1)

# ---------------------------------
# Embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Failed to initialize embeddings: {e}")
    raise

# ---------------------------------
# Load FAISS
try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    logging.info("FAISS vectorstore loaded successfully")
except Exception as e:
    logging.error(f"Failed to load FAISS vectorstore from '{DB_FAISS_PATH}': {e}")
    raise

# ---------------------------------
# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a smart, concise, and friendly AI assistant.
Use the context below to answer naturally and clearly, like a human.
If the context doesn't have the answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------------
# Load LLM (Offline models in ./models first, then Ollama)
def load_llm():
    # Helper to select torch dtype & device for CPU-safe operation
    def preferred_dtype_and_device():
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            return torch.float16, "cuda"
        return torch.float32, "cpu"

    dtype, device = preferred_dtype_and_device()

    # Try offline models located under ./models/<name>
    offline_candidates = ["phi-2", "tinyllama"]
    for model_name in offline_candidates:
        model_path = os.path.join("models", model_name)
        if os.path.exists(model_path):
            try:
                logging.info(f"Attempting offline model at: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                # Load model in a CPU-friendly way (avoid device_map="auto" on CPU-only machines)
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
                if device == "cpu":
                    model.to("cpu")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == "cuda" else -1,
                    max_new_tokens=200,
                    temperature=0.6
                )
                logging.info(f"Using offline model: {model_name}")
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logging.warning(f"Offline model '{model_name}' failed to load: {e}")

    # Fallback to Ollama (local Ollama server)
    try:
        logging.info("Attempting Ollama (local) fallback...")
        llm = Ollama(model="phi", base_url="http://127.0.0.1:11435")
        # quick sanity check
        _ = llm.invoke("Hello")
        logging.info("Using Ollama Local LLM")
        return llm
    except Exception as e:
        logging.warning(f"Ollama fallback failed or is unavailable: {e}")

    # If all fails — raise an informative error
    raise RuntimeError("No LLM could be loaded. Provide an offline model under ./models or run Ollama locally.")

# Load LLM (this may raise if no model available)
llm = load_llm()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# ---------------------------------
# Flask API Backend
app = Flask(__name__)

# ---------------------------------
# Chat history and logging
chat_history = []

def log_interaction(question, answer):
    chat_history.append({"question": question, "answer": answer})
    try:
        with open("logs.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), question, answer])
    except Exception as e:
        logging.warning(f"Failed to write logs.csv: {e}")

# ---------------------------------
# /ask endpoint
@app.post("/ask")
def ask():
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"answer": "Please enter a question."}), 400

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except Exception:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        response = response.strip()
        log_interaction(question, response)
        return jsonify({"answer": response})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}"}), 500

# ---------------------------------
# /chat endpoint for Streamlit
@app.post("/chat")
def chat():
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"answer": "Please enter a question.", "history": chat_history}), 400

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except Exception:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        response = response.strip()
        log_interaction(question, response)
        return jsonify({"answer": response, "history": chat_history})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}", "history": chat_history}), 500

# ---------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Backend running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
