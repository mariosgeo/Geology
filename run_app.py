"""
Launcher script for the Geology Streamlit Application.
Usage:
    python run_app.py
or:
    streamlit run app.py
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "streamlit", "run", app_path] + sys.argv[1:]
    print(f"Launching Streamlit app: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)
