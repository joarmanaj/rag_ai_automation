# streamlit_ui.py
import streamlit as st
import requests

# -----------------------------
# Replace this with your deployed Render Flask backend URL after deployment
API_URL = "https://<your-flask-service>.onrender.com/chat"

# -----------------------------
st.set_page_config(page_title="RAG AI Assistant", layout="wide")
st.title("🤖 RAG AI Assistant")
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
                    API_URL,
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
