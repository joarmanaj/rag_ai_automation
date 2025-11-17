import os, sys, time, subprocess, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FOLDERS = ["docs", "data"]
INGEST_SCRIPT = "ingest.py"
UI_SCRIPT = "ui.py"
CHECK_INTERVAL = 10
UI_PORT = 7860

if "--port" in sys.argv:
    try:
        UI_PORT = int(sys.argv[sys.argv.index("--port")+1])
    except: pass

def get_latest_mod_time():
    latest = 0
    for folder in FOLDERS:
        if not os.path.exists(folder): continue
        for root, _, files in os.walk(folder):
            for f in files:
                latest = max(latest, os.path.getmtime(os.path.join(root,f)))
    return latest

def run_ui():
    logging.info(f"🚀 Launching UI on port {UI_PORT}...")
    return subprocess.Popen([sys.executable, UI_SCRIPT, "--port", str(UI_PORT)])

def run_ingest():
    logging.info("⚠️ Detected changes. Running ingest.py...")
    result = subprocess.run([sys.executable, INGEST_SCRIPT], capture_output=True, text=True)
    if result.returncode == 0: logging.info("✅ Ingest completed.")
    else: logging.error(f"❌ Ingest failed:\n{result.stderr}")

if __name__ == "__main__":
    ui_proc = run_ui()
    last_mod_time = get_latest_mod_time()
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            current_mod_time = get_latest_mod_time()
            if current_mod_time > last_mod_time:
                logging.info("⚠️ Changes detected. Updating vectorstore...")
                run_ingest()
                logging.info("🔄 Restarting UI...")
                ui_proc.terminate()
                ui_proc.wait()
                ui_proc = run_ui()
                last_mod_time = current_mod_time
    except KeyboardInterrupt:
        logging.info("🛑 Stopping watch process...")
        ui_proc.terminate()
        ui_proc.wait()
