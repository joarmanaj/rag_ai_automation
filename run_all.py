# run_all.py
# ----------------------------------------
import subprocess
import time
import sys
import os
import logging
import webbrowser

READY_FLAG = "main_ready.flag"
VECTORSTORE_PATH = "vectorstore.faiss"
INGEST_SCRIPT = "ingest.py"
WATCH_SCRIPT = "watch.py"
UI_SCRIPT = "ui.py"
BACKEND_SCRIPT = "main.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------
def run_script(script_name, args=None):
    """Run a Python script in the same environment."""
    cmd = [sys.executable, script_name]
    if args:
        cmd += args
    logging.info(f"🚀 Launching {script_name} ...")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# ----------------------------------------
def wait_for_flag(flag_path, timeout=60):
    """Wait until a file exists or timeout."""
    logging.info(f"⏳ Waiting for backend to be ready (flag: {flag_path}) ...")
    start = time.time()
    while not os.path.exists(flag_path):
        if time.time() - start > timeout:
            logging.error(f"❌ Timeout waiting for {flag_path}")
            return False
        time.sleep(1)
    logging.info("✅ Backend is ready!")
    return True

# ----------------------------------------
def needs_ingest():
    """Check if ingest.py should run based on vectorstore existence or updated documents."""
    if not os.path.exists(VECTORSTORE_PATH):
        logging.warning(f"⚠️ Vectorstore not found at '{VECTORSTORE_PATH}'")
        return True

    vectorstore_mtime = os.path.getmtime(VECTORSTORE_PATH)
    for folder in ["docs", "data"]:
        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                if os.path.getmtime(path) > vectorstore_mtime:
                    logging.warning(f"⚠️ New/updated document detected: {path}")
                    return True
    return False

# ----------------------------------------
def terminate_process(proc):
    if proc and proc.poll() is None:
        logging.info(f"🛑 Terminating {proc.args[1]} ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ----------------------------------------
if __name__ == "__main__":
    logging.info(f"🧠 Using Python from: {sys.executable}")

    # Default UI port
    UI_PORT = "7860"
    if "--port" in sys.argv:
        try:
            UI_PORT = sys.argv[sys.argv.index("--port") + 1]
        except Exception:
            logging.warning(f"Invalid port argument, using default {UI_PORT}")

    # Step 0: Run ingest.py if needed
    if needs_ingest():
        logging.info("⚠️ Running ingest.py to update vectorstore...")
        ingest_proc = run_script(INGEST_SCRIPT)
        ingest_proc.wait()
        logging.info("✅ Ingest completed. Vectorstore is ready.")

    # Step 1: Start watcher
    watcher = run_script(WATCH_SCRIPT)
    logging.info("👀 Watcher started to monitor docs/ and data/")

    # Step 2: Start backend
    backend = run_script(BACKEND_SCRIPT)

    # Step 3: Wait for main_ready.flag
    if wait_for_flag(READY_FLAG):
        # Step 4: Launch UI with port
        ui = run_script(UI_SCRIPT, args=["--port", UI_PORT])
        logging.info(f"✅ Backend and UI are running on port {UI_PORT}.")

        # Automatically open browser
        url = f"http://127.0.0.1:{UI_PORT}"
        logging.info(f"🌐 Opening browser at {url}")
        webbrowser.open(url)

        try:
            # Keep script alive while all processes run
            while True:
                if backend.poll() is not None:
                    logging.error("❌ Backend exited unexpectedly. Shutting down UI and watcher.")
                    terminate_process(ui)
                    terminate_process(watcher)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Stopping all processes due to KeyboardInterrupt...")
            terminate_process(backend)
            terminate_process(ui)
            terminate_process(watcher)
    else:
        logging.error("❌ Backend failed to initialize. Exiting.")
        terminate_process(backend)
        terminate_process(watcher)
