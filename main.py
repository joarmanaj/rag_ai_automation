
import os
import logging
import warnings
import torch
import atexit
import traceback
import gradio as gr

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate

# -------------------------------
# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Paths
DB_FAISS_PATH = "vectorstore.faiss"
READY_FLAG = "main_ready.flag"

if not os.path.exists(DB_FAISS_PATH):
    logging.error(f"FAISS database not found at '{DB_FAISS_PATH}'. Run ingest.py first!")
    exit()

# -------------------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Load FAISS
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
logging.info("FAISS vectorstore loaded successfully")

# -------------------------------
# ✅ Signal backend ready
try:
    with open(READY_FLAG, "w") as f:
        f.write("ready")
    logging.info("✅ main_ready.flag created successfully — backend initialized.")
except Exception as e:
    logging.error(f"❌ Could not create main_ready.flag: {e}")

# -------------------------------
# Remove ready flag on exit
def remove_ready_flag():
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
        logging.info("🧹 main_ready.flag removed on exit.")

atexit.register(remove_ready_flag)

# -------------------------------
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

# -------------------------------
# Load LLM (Ollama → TinyLlama fallback)
def load_llm():
    try:
        logging.info("Trying Ollama Local...")
        llm = Ollama(model="phi", base_url="http://127.0.0.1:11435")
        _ = llm.invoke("Hello")  # sanity check
        print("\n🤖 Using Ollama Local LLM\n")
        logging.info("Ollama Local loaded successfully")
        return llm
    except Exception as e:
        logging.warning(f"Ollama Local failed: {e}")

    try:
        logging.info("Loading TinyLlama 1.1B from Hugging Face...")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            device_map="auto"
        )
        print("\n🤖 Ollama not available. Using TinyLlama 1.1B Chat model\n")
        logging.info("TinyLlama loaded successfully")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logging.error(f"Failed to load TinyLlama: {e}")
        exit("❌ No LLM available. Exiting...")

# -------------------------------
# Initialize Lazy LLM + RetrievalQA
llm = None
qa_chain = None

def ensure_chain():
    """Load LLM and RetrievalQA chain only once."""
    global llm, qa_chain
    if llm is None:
        llm = load_llm()
    if qa_chain is None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False
        )
    return qa_chain

# -------------------------------
# Gradio function
def answer_question(user_query):
    try:
        if not user_query.strip():
            return "⚠️ Please enter a question."

        qa = ensure_chain()
        docs = retriever.get_relevant_documents(user_query)
        context_text = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context_text, question=user_query)

        # Run query
        try:
            response = qa.run(final_prompt)
        except Exception:
            result = qa.invoke({"query": final_prompt})
            response = result.get("result", str(result))

        return response.strip() if response else "No answer generated."
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error: {e}\n{tb}")
        return f"❌ Error:\n{tb}"

# -------------------------------
# Gradio UI
css = """
#chatbox { max-width: 900px; margin: auto; }
"""

with gr.Blocks(css=css, title="RAG AI Assistant") as demo:
    gr.Markdown("## 🤖 RAG AI Assistant\nAsk questions about your ingested documents.")
    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="Type a question...", lines=2)
        submit_btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=12)
    submit_btn.click(fn=answer_question, inputs=query, outputs=answer)
    query.submit(fn=answer_question, inputs=query, outputs=answer)

print("🚀 Launching Gradio UI...")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
