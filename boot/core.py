import sys
import subprocess
import importlib
from pathlib import Path
import myFunc.log as lg

TKINTER_REQUIRED = 8.6
PYTHON_REQUIRED = (3, 13, 3)
USE_EMOJI = True

def in_venv():
    return sys.prefix != sys.base_prefix

def get_required_modules(req_file="requirements.txt"):
    # Cerca requirements.txt nella directory dove si trova questo file (boot/core.py)
    project_root = Path(__file__).parent
    path = project_root / req_file
    print(f"Looking for requirements.txt at: {path.resolve()}")
    if not path.exists():
        sys.exit(f"File '{req_file}' not found.")
    modules = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mod = line.split("==")[0].lower()
            if mod == "pillow":
                mod = "PIL"
            elif mod == "opencv-python":
                mod = "cv2"
            elif mod == "pywavelets":
                mod = "pywt"
            elif mod == "scikit-image":
                mod = "skimage"
            elif mod == "packaging":
                mod = "packaging"
            elif mod == "pytest":
                mod = "pytest"
            modules.append(mod)
    return modules

def install_missing_libraries(req_file="requirements.txt"):
    missing = []
    for mod in get_required_modules(req_file):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        lg.log("Installing missing libraries: " + ", ".join(missing), "📦")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            lg.log("Installation completed.", "✅")
        except subprocess.CalledProcessError:
            sys.exit("An error occurred while installing the required packages.")
    else:
        lg.log("All required libraries are already installed.", "🔎")

def main(use_emoji):
    global USE_EMOJI
    if not use_emoji:
        USE_EMOJI = False
        lg.main(use_emoji=USE_EMOJI)

    if not in_venv():
        sys.exit("This must be run inside a virtual environment.")
    lg.log("Venv detected!", "📂")

    install_missing_libraries()

    try:
        lg.log("Launching application...", "🚀")
        subprocess.check_call([sys.executable, "boot/app.py"])
    except subprocess.CalledProcessError:
        sys.exit("An error occurred while launching the application.")

# Supporta sia importazione che esecuzione diretta
if __name__ == "__main__":
    main(USE_EMOJI)
