import os
import logging
import warnings
import torch
import traceback
from flask import Flask, request, jsonify

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

if not os.path.exists(DB_FAISS_PATH):
    logging.error(f"FAISS database not found at '{DB_FAISS_PATH}'")
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
# Prompt
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
# Load LLM
def load_llm():
    try:
        logging.info("Trying Ollama...")
        llm = Ollama(model="phi", base_url="http://127.0.0.1:11435")
        _ = llm.invoke("Hello")
        logging.info("Using Ollama Local LLM")
        return llm
    except Exception as e:
        logging.warning(f"Ollama failed: {e}")

    try:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.6
        )
        logging.info("Using TinyLlama fallback")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logging.error(f"TinyLlama failed: {e}")
        exit("No LLM available.")

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

# Simple /ask endpoint
@app.post("/ask")
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question.strip():
            return jsonify({"answer": "⚠️ Please enter a question."})

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        return jsonify({"answer": response.strip()})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}"})


# Chat-like endpoint for Streamlit with history
chat_history = []

@app.post("/chat")
def chat():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "⚠️ Please enter a question.", "history": chat_history})

        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=question)

        try:
            response = qa_chain.run(final_prompt)
        except:
            result = qa_chain.invoke({"query": final_prompt})
            response = result.get("result", "")

        response = response.strip()
        chat_history.append({"question": question, "answer": response})

        return jsonify({"answer": response, "history": chat_history})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"answer": f"Error:\n{tb}", "history": chat_history})


if __name__ == "__main__":
    logging.info("🚀 Backend running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
