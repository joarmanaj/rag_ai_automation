# streamlit_ui.py
import streamlit as st
import requests
import webbrowser
import time

# -----------------------------
# Ngrok URL (dynamic display)
public_url = "Ngrok tunnel not active"
try:
    from pyngrok import conf, ngrok
    conf.get_default().ngrok_path = r"C:\Users\HP\RAG_AI_AUTOMATION\ngrok.exe"
    tunnels = ngrok.get_tunnels()
    if tunnels:
        public_url = tunnels[0].public_url
        # Open the Ngrok URL in browser (only once)
        webbrowser.open(public_url)
        time.sleep(1)  # brief pause to ensure browser opens
except Exception as e:
    public_url = f"Ngrok error: {e}"

# -----------------------------
st.set_page_config(page_title="RAG AI Assistant", layout="wide")
st.title("🤖 RAG AI Assistant")
st.markdown(f"**Public Ngrok URL:** {public_url}")
st.markdown("Ask questions about your ingested documents.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
question = st.text_input("Your Question", placeholder="Type a question and press Enter")

# Send question
if st.button("Ask") or question:
    if question.strip() != "":
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:5000/chat",
                    json={"question": question}
                )
                data = res.json()
                st.session_state.chat_history = data.get("history", [])
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
        question = ""  # Clear input after sending

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**AI:** {chat['answer']}")
    st.markdown("---")
