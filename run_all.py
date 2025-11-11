import subprocess
import time
import sys
import os

READY_FLAG = "main_ready.flag"

def run_script(script_name):
    """Run a Python script in the same environment."""
    print(f"🚀 Launching {script_name} ...")
    return subprocess.Popen([sys.executable, script_name])

def wait_for_flag(flag_path, timeout=60):
    """Wait until a file exists or timeout."""
    print(f"⏳ Waiting for backend to be ready (flag: {flag_path}) ...")
    start = time.time()
    while not os.path.exists(flag_path):
        if time.time() - start > timeout:
            print(f"❌ Timeout waiting for {flag_path}")
            return False
        time.sleep(1)
    print("✅ Backend is ready!")
    return True

if __name__ == "__main__":
    # Ensure correct virtual environment
    print(f"🧠 Using Python from: {sys.executable}")

    # Step 1: Start backend (main.py)
    backend = run_script("main.py")

    # Step 2: Wait for main_ready.flag
    if wait_for_flag(READY_FLAG):
        # Step 3: Launch UI
        ui = run_script("ui.py")
        print("✅ Both backend and UI are running.")
        print("🌐 Open http://127.0.0.1:7860 in your browser to use the RAG App.")

        try:
            # Keep script alive while both processes run
            backend.wait()
            ui.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping all processes...")
            backend.terminate()
            ui.terminate()
    else:
        print("❌ Backend failed to initialize. Exiting.")
        backend.terminate()
