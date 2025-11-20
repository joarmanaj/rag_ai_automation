import subprocess
import os
import time
import sys

# -------------------------------
# Paths (update if different)
VENV_PATH = r"C:\Users\HP\RAG_AI_AUTOMATION\venv_rag\Scripts\activate.bat"
FLASK_APP = r"C:\Users\HP\RAG_AI_AUTOMATION\main.py"
NGROK_EXE = r"C:\Users\HP\RAG_AI_AUTOMATION\ngrok.exe"
PORT = 5000

# -------------------------------
# Helper to run a command in a new terminal
def run_in_new_terminal(cmd):
    subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)

# -------------------------------
# Activate venv and run Flask
flask_cmd = f'call "{VENV_PATH}" && python "{FLASK_APP}"'
run_in_new_terminal(flask_cmd)

# Give Flask a few seconds to start
time.sleep(3)

# Run ngrok tunnel for the same port
ngrok_cmd = f'call "{VENV_PATH}" && "{NGROK_EXE}" http {PORT}'
run_in_new_terminal(ngrok_cmd)

print(" Flask + ngrok launched in separate terminals")
