import os
import logging
import warnings
import torch
import gradio as gr
import threading
import time
import traceback

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_FAISS_PATH = "vectorstore.faiss"
if not os.path.exists(DB_FAISS_PATH):
    logging.error(f"FAISS database not found at '{DB_FAISS_PATH}'. Run ingest.py first!")
    exit()

# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = None
retriever = None
llm = None
pipeline_ready = False

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a friendly and intelligent assistant.
Use the context below to answer clearly and conversationally.
If the context does not contain enough information, say so politely.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
def load_pipeline():
    """Load FAISS vectorstore and LLM (public, CPU-friendly)."""
    global db, retriever, llm, pipeline_ready

    try:
        logging.info("Loading FAISS vectorstore...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        logging.info("FAISS loaded successfully ✅")
    except Exception as e:
        logging.error(f"Error loading FAISS: {e}")
        exit("Cannot proceed without FAISS.")

    try:
        model_id = "google/flan-t5-small"
        logging.info(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("LLM loaded successfully ✅")
    except Exception as e2:
        logging.error(f"Failed to load model: {e2}")
        exit("No LLM available. Exiting...")

    pipeline_ready = True
    logging.info("Pipeline fully loaded and ready! 🚀")

# -------------------------------
def chat(query, chat_history):
    global llm, retriever, pipeline_ready

    if not query.strip():
        return chat_history

    if not pipeline_ready:
        chat_history.append({"role": "assistant", "content": "⏳ Please wait, model is still loading..."})
        return chat_history

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": "Thinking... 💭"})
    yield chat_history

    try:
        docs = retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in docs])
        final_prompt = prompt_template.format(context=context_text, question=query)

        outputs = llm.pipeline(final_prompt)
        answer = outputs[0]["generated_text"].strip()

        simulated = ""
        for token in answer.split():
            simulated += token + " "
            chat_history[-1]["content"] = simulated.strip()
            yield chat_history
            time.sleep(0.03)

        chat_history[-1]["content"] = simulated.strip()
        yield chat_history

    except Exception as e:
        traceback.print_exc()
        chat_history[-1]["content"] = f"❌ Error: {str(e)}"
        yield chat_history

# -------------------------------
with gr.Blocks(auth=None, analytics_enabled=False, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align:center; color:#4B0082;'>🧠 RAG AI Assistant</h1>")
    gr.Markdown("<p style='text-align:center;'>Ask your dataset anything — it retrieves and explains clearly.</p>")

    with gr.Row():
        with gr.Column(scale=3):
            chat_display = gr.Chatbot(label="Chat", type="messages")
            msg = gr.Textbox(label="Your Question:", placeholder="Type your question here...")
            clear = gr.Button("Clear")

            msg.submit(chat, [msg, chat_display], [chat_display], queue=False)
            clear.click(lambda: [], [], chat_display)

        with gr.Column(scale=1):
            gr.Markdown("### 💡 How to Use")
            gr.Markdown("""
- Ask questions related to your dataset.  
- The AI retrieves context and responds conversationally.  
- First response may take a few seconds to load.  
- Click **Clear** to start a new session.
            """)

    gr.Markdown("<p style='text-align:center; color:#888;'>Built with LangChain + FAISS + HuggingFace</p>")

# Start loading pipeline in the background
threading.Thread(target=load_pipeline, daemon=True).start()

# Launch without login requirement
demo.launch()
