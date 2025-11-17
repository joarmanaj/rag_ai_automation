# ui.py
import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/ask"   # Our backend endpoint in main.py

def ask_backend(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        return response.json().get("answer", "Error: no answer returned.")
    except Exception as e:
        return f"❌ UI Error: {str(e)}"

gr.Interface(
    fn=ask_backend,
    inputs=gr.Textbox(label="Ask a question about your documents"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG AI Assistant",
    description="Ask questions and get answers retrieved from your documents."
).launch(server_port=7860)
