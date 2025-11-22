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
    logging.info("FAISS not found. Running ingest.py...")
    try:
        subprocess.run(["python", "ingest.py"], check=True)
        logging.info("ingest.py completed.")
    except Exception as e:
        logging.error(f"ingest.py failed: {e}")

# ---------------------------------
# Embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Embedding init failed: {e}")
    raise

# ---------------------------------
# Load FAISS
try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    logging.info("FAISS loaded successfully.")
except Exception as e:
    logging.error(f"FAISS load error: {e}")
    raise

# ---------------------------------
# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a smart, concise, friendly AI assistant.
Use the context below to answer clearly.
If the context does not contain the answer, say so.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------------
# Load LLM (offline first, then Ollama)
def load_llm():
    try_models = ["phi-2", "tinyllama", "tinyllama_new"]

    for model_name in try_models:
        model_path = os.path.join("models", model_name)
        if os.path.exists(model_path):
            try:
                logging.info(f"Loading offline model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                    max_new_tokens=200,
                    temperature=0.6
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logging.warning(f"Offline model failed: {e}")

    # No offline model → TRY OLLAMA
    try:
        logging.info("Trying Ollama fallback...")
        llm = Ollama(model="phi", base_url="http://127.0.0.1:11435")
        llm.invoke("Hello")
        return llm
    except Exception:
        logging.warning("Ollama not available.")

    # Nothing works → fail clearly
    raise RuntimeError(
        "No LLM available. Upload an offline model into /models OR run Ollama locally."
    )

llm = load_llm()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# ---------------------------------
# Flask Backend
app = Flask(__name__)

chat_history = []

def log_interaction(q, a):
    chat_history.append({"question": q, "answer": a})
    try:
        with open("logs.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now(), q, a])
    except Exception as e:
        logging.warning(f"logs.csv write failed: {e}")

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
        context = "\n".join([d.page_content for d in docs])

        final_prompt = prompt_template.format(context=context, question=question)

        try:
            answer = qa_chain.run(final_prompt)
        except Exception:
            result = qa_chain.invoke({"query": final_prompt})
            answer = result.get("result", "")

        answer = answer.strip()
        log_interaction(question, answer)

        return jsonify({"answer": answer})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}"}), 500

# ---------------------------------
# /chat endpoint
@app.post("/chat")
def chat():
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please enter a question.", "history": chat_history}), 400

        docs = retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in docs])

        final_prompt = prompt_template.format(context=context, question=question)

        try:
            answer = qa_chain.run(final_prompt)
        except Exception:
            result = qa_chain.invoke({"query": final_prompt})
            answer = result.get("result", "")

        answer = answer.strip()
        log_interaction(question, answer)

        return jsonify({"answer": answer, "history": chat_history})
    except Exception:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}", "history": chat_history}), 500

