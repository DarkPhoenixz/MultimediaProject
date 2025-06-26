import os
import sys
import subprocess
import platform
import myFunc.log as lg

VENV_DIR = "venv"
PYTHON_REQUIRED = "3.13.3"
PYTHON_EXEC = "python3.13"

USE_EMOJI = True
def parse_flags():
    global USE_EMOJI
    if "--no-emoji" in sys.argv:
        USE_EMOJI = False
    elif os.environ.get("NO_EMOJI") == "1":
        USE_EMOJI = False
    # Fallback automatico per Windows
    if os.name == "nt":
        try:
            print("on Windows")
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            USE_EMOJI = False
    lg.main(use_emoji = USE_EMOJI)

def in_venv():
    return sys.prefix != sys.base_prefix


def get_python_version():
    return platform.python_version()


def check_tkinter_version(min_version=8.6):
    try:
        import tkinter
        tk_version = float(f"{tkinter.TkVersion:.2f}")
        if tk_version < min_version:
            
            lg.log(f"Tkinter version too old: {tk_version} < {min_version}", "⚠️")
            return False
        return True
    except ImportError:
        lg.log("Tkinter not found.", "❌")
        return False


def create_venv():
    lg.log(f"Creating virtual environment using {PYTHON_EXEC}...", "📦")
    subprocess.check_call([PYTHON_EXEC, "-m", "venv", VENV_DIR])


def run_inside_venv():
    python_path = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
    boot_file = os.path.abspath(__file__)  # Percorso assoluto al file boot.py
    lg.log(f"Restarting inside virtual environment using {python_path}", "🔁")
    args = [python_path, boot_file]
    if not USE_EMOJI:
        args.append("--no-emoji")
    subprocess.check_call(args)
    sys.exit()






def main():
    parse_flags()
    current_version = get_python_version()

    if current_version != PYTHON_REQUIRED or not in_venv():
        lg.log(f"Required Python version: {PYTHON_REQUIRED}, current: {current_version}", "🐍")
        if not os.path.isdir(VENV_DIR):
            create_venv()
        run_inside_venv()
        return

    if not check_tkinter_version():
        sys.exit("Tkinter check failed. Please install or recompile Python with Tk support.")
    lg.log("Environment is correctly set up. Launching core.py", "✅")


    import core
    core.main(use_emoji=USE_EMOJI)


if __name__ == "__main__":
    main()
