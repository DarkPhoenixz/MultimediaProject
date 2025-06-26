import os
import sys
import subprocess
import platform
import argparse
import shutil
from pathlib import Path
import myFunc.log as lg

class EnvironmentSetupError(Exception):
    """Custom exception for environment setup issues."""
    pass

VENV_DIR = Path("venv")
PYTHON_REQUIRED = "3.13.3"
USE_EMOJI = True

def parse_flags() -> argparse.Namespace:
    """Parses command-line arguments and sets global flags."""
    global USE_EMOJI
    parser = argparse.ArgumentParser(description="Boot script for the application.")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in logs.")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate the venv.")
    args = parser.parse_args()

    if args.no_emoji or os.environ.get("NO_EMOJI") == "1":
        USE_EMOJI = False

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            lg.log("Console output encoding set to UTF-8 for Windows.", "ℹ️")
        except Exception as e:
            lg.log(f"Failed to set console output encoding: {e}", "⚠️")
            USE_EMOJI = False

    lg.main(use_emoji=USE_EMOJI)
    return args

def in_venv() -> bool:
    return sys.prefix != sys.base_prefix

Version = None
if in_venv():
    try:
        from packaging.version import Version as _Version
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
        from packaging.version import Version as _Version
    Version = _Version

def get_python_version() -> str:
    return platform.python_version()

def check_python_requirement() -> None:
    current = Version(get_python_version())
    req = Version(PYTHON_REQUIRED)
    lg.log(f"Python version {current} found, required: {req}", "✅")
    if current < req:
        raise EnvironmentSetupError(f"Requires Python >= {PYTHON_REQUIRED}, current: {current}")

def check_tkinter_version(min_version: float = 8.6) -> None:
    try:
        import tkinter
        tk_version = float(f"{tkinter.TkVersion:.2f}")
        if tk_version < min_version:
            raise EnvironmentSetupError(f"Tkinter too old: {tk_version} < {min_version}")
        lg.log(f"Tkinter version {tk_version} found, required: {min_version}.", "✅")
    except ImportError:
        raise EnvironmentSetupError("Tkinter not found. Install Python with Tk support.")
    except Exception as e:
        raise EnvironmentSetupError(f"Unexpected error checking Tkinter: {e}")

def create_platform_file() -> None:
    with (VENV_DIR / "platform.txt").open("w") as f:
        f.write(platform.system().lower())

def check_venv_platform() -> None:
    platform_file = VENV_DIR / "platform.txt"
    if not platform_file.exists():
        return
    recorded = platform_file.read_text().strip().lower()
    current = platform.system().lower()
    lg.log(f"Virtual environment platform consolidated to {recorded}", "✅")
    if recorded != current:
        raise EnvironmentSetupError(f"Venv created on {recorded}, now running on {current} – incompatible.")

def create_venv() -> None:
    if VENV_DIR.exists():
        lg.log(f"Virtual environment exists at {VENV_DIR}", "✅")
        return
    lg.log(f"Creating virtual environment...", "📦")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        create_platform_file()
        lg.log("Virtual environment created.", "✅")
    except Exception as e:
        raise EnvironmentSetupError(f"Failed creating venv: {e}")

def run_inside_venv() -> None:
    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    python_path = VENV_DIR / scripts / exe
    boot_file = Path(__file__).resolve()

    if not python_path.exists():
        raise EnvironmentSetupError(f"Cannot find Python in venv at {python_path}")

    lg.log(f"Restarting inside venv using {python_path}", "🔁")
    args = [str(python_path), str(boot_file)]
    if not USE_EMOJI:
        args.append("--no-emoji")

    lg.log(f"Command: {args}", "ℹ️")
    process = subprocess.Popen(args)
    process.wait()
    sys.exit(process.returncode)


def bootstrap() -> None:
    args = parse_flags()

    try:

        if args.force_recreate and VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            lg.log("Removed existing venv (--force-recreate)", "♻️")

        if not in_venv() or not VENV_DIR.exists():
            lg.log("Setting up environment...", "🐍")
            create_venv()
            run_inside_venv()
            return
        
        check_venv_platform()
        check_python_requirement()
        check_tkinter_version()

        lg.log("Environment OK. Launching core...", "✅")
        import core
        if not hasattr(core, "main"):
            raise EnvironmentSetupError("core.py must define a `main(use_emoji)` function.")
        core.main(use_emoji=USE_EMOJI)

    except EnvironmentSetupError as e:
        lg.log(f"Error: {e}", "❌")
        sys.exit(1)
    except ImportError as e:
        lg.log(f"Import error: {e}", "❌")
        sys.exit(1)
    except Exception as e:
        lg.log(f"Unexpected error: {e}", "❌")
        sys.exit(1)

if __name__ == "__main__":
    bootstrap()
