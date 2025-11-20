import os
import warnings
import torch
import traceback
import subprocess
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
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
        subprocess.run(["python", "ingest.py"], check=True)
        logging.info(" ingest.py completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f" ingest.py failed: {e}")
        exit(1)

# ---------------------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------
# Load FAISS
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
logging.info("FAISS vectorstore loaded successfully")

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
# Load LLM (Phi-2 offline first, then TinyLlama, then Ollama)
def load_llm():
    # Attempt Phi-2 offline
    try:
        model_path = r"C:\Users\HP\RAG_AI_AUTOMATION\models\phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.6
        )
        logging.info(" Using Phi-2 offline")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logging.warning(f"Phi-2 offline failed: {e}")

    # Attempt TinyLlama offline
    try:
        model_path = r"C:\Users\HP\RAG_AI_AUTOMATION\models\tinyllama"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.6
        )
        logging.info(" Using TinyLlama offline")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logging.warning(f"TinyLlama offline failed: {e}")

    # Fallback to Ollama
    try:
        logging.info("Trying Ollama...")
        llm = Ollama(model="phi", base_url="http://127.0.0.1:11435")
        _ = llm.invoke("Hello")
        logging.info(" Using Ollama Local LLM")
        return llm
    except Exception as e:
        logging.error(f"No LLM available: {e}")
        exit("No LLM could be loaded.")

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
# Ngrok setup
try:
    conf.get_default().ngrok_path = r"C:\Users\HP\RAG_AI_AUTOMATION\ngrok.exe"
    public_url = ngrok.connect(5000)
    logging.info(f" Public API URL: {public_url}")
except Exception as e:
    logging.warning(f"Ngrok tunnel failed: {e}")
    public_url = "http://127.0.0.1:5000"

# ---------------------------------
# Chat history and logging
chat_history = []

def log_interaction(question, answer):
    chat_history.append({"question": question, "answer": answer})
    with open("logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), question, answer])

# ---------------------------------
# /ask endpoint
@app.post("/ask")
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"answer": " Please enter a question."})

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        response = response.strip()
        log_interaction(question, response)
        return jsonify({"answer": response})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}"})

# ---------------------------------
# /chat endpoint for Streamlit
@app.post("/chat")
def chat():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"answer": " Please enter a question.", "history": chat_history})

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        response = response.strip()
        log_interaction(question, response)
        return jsonify({"answer": response, "history": chat_history})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}", "history": chat_history})

# ---------------------------------
if __name__ == "__main__":
    logging.info(f" Backend running at {public_url}")
    app.run(host="0.0.0.0", port=5000)
