import os
import sys
import subprocess
import venv
from pathlib import Path

def main():
    print("🚀 StressTracker Setup & Launcher")
    print("================================")
    
    # 1. Detect/Create Venv
    venv_dir = Path(".venv")
    created_new = False
    
    if not venv_dir.exists():
        print(f"📦 Creating virtual environment at {venv_dir}...")
        try:
            venv.create(venv_dir, with_pip=True)
            created_new = True
            print("✅ Virtual environment created.")
        except Exception as e:
            print(f"❌ Failed to create venv: {e}")
            sys.exit(1)
    else:
        print("✅ Found existing virtual environment.")
    
    # 2. Determine Python Executable
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else: # macOS / Linux
        python_exe = venv_dir / "bin" / "python"
        
    if not python_exe.exists():
         print(f"❌ Error: Python executable not found at {python_exe}")
         print("   Please delete the .venv folder and try again.")
         sys.exit(1)

    # 3. Install Dependencies
    # If we just created the venv, or if requirements changed, we should install.
    # For simplicity in this script, we ensure install every time (pip is fast if cached).
    print("⬇️  Checking/Installing dependencies (this may take a moment)...")
    try:
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
        print("✅ Dependencies installed.")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")
        sys.exit(1)

    # 4. Check & Setup Ollama
    print("🦙 Checking AI Model Status...")
    try:
        # Check if ollama is installed
        subprocess.check_call(["ollama", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("   Ollama found. Pulling/Verifying model 'llama3.2'...")
        subprocess.check_call(["ollama", "pull", "llama3.2"])
        print("✅ Model ready.")
        
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  Warning: 'ollama' is not installed or not in PATH.")
        print("   The AI analysis features will not work without it.")
        print("   Please install it from https://ollama.com/download")
        # We don't exit here, we let the app run (it handles missing LLM gracefully mostly)
        input("   Press Enter to continue anyway (or Ctrl+C to abort)...")

    # 5. Run Streamlit
    print("🧠 Starting StressTracker AI...")
    print("   (Press Ctrl+C to stop)")
    print("--------------------------------")
    
    cmd = [str(python_exe), "-m", "streamlit", "run", "app.py"]
    try:
        subprocess.check_call(cmd)
    except KeyboardInterrupt:
        print("\n👋 StressTracker stopped. Have a stress-free day!")
    except subprocess.CalledProcessError as e:
        # Streamlit returning non-zero is common on forced exit, usually fine
        if e.returncode != 0:
            pass 

if __name__ == "__main__":
    main()
