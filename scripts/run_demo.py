"""CryptoFraud Interactive Demo Launcher.

Starts the FastAPI inference server with XGBoost and SHAP explainability,
polls for healthy startup, and automatically opens the interactive AML Dashboard in your browser.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
PORT = 8000
DASHBOARD_URL = f"http://{HOST}:{PORT}/dashboard"
HEALTH_URL = f"http://{HOST}:{PORT}/health"


def wait_for_server(max_retries: int = 30, delay: float = 0.5) -> bool:
    """Poll the /health endpoint until the server is ready."""
    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(delay)
    return False


def main():
    print("=" * 70)
    print(" 🚀 Launching CryptoFraud Bitcoin AML & SHAP Explainability Demo")
    print("=" * 70)
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Server target:     http://{HOST}:{PORT}")
    print(f"Dashboard UI:      {DASHBOARD_URL}")
    print("-" * 70)

    # Use the active virtual environment python if available
    python_bin = sys.executable

    cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "service.main:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    print("Starting FastAPI Uvicorn service...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    try:
        print("Waiting for model and SHAP explainer to initialize...")
        if wait_for_server():
            print("\n" + "✓" * 70)
            print(" [SUCCESS] CryptoFraud Service is HEALTHY and READY!")
            print(f" Opening dashboard in default browser: {DASHBOARD_URL}")
            print("✓" * 70 + "\n")
            webbrowser.open(DASHBOARD_URL)
        else:
            print("[WARNING] Server took longer than expected to start.")
            print(f"You can manually visit: {DASHBOARD_URL}")

        print("Press Ctrl+C at any time to shut down the demo server.\n")
        proc.wait()

    except KeyboardInterrupt:
        print("\nStopping demo server gracefully...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped. Have a great day!")


if __name__ == "__main__":
    main()
