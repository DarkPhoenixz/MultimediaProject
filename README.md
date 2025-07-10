# MultimediaProject

<p align="center">
  <img src="githubDecors/logo.png" width="120" alt="Project Logo"/>
</p>

<p align="center">
  <a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python Version"></a>
  <img src="https://img.shields.io/badge/tkinter-8.6%2B-blueviolet" alt="Tkinter Version">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform">
</p>

<p align="center">
  <b>Powerful, modular toolkit for image steganography and watermarking.<br/>
  Featuring DCT, DFT, DWT, LSB, DSSS algorithms and a modern GUI.</b>
</p>

---

## 🚀 Demo

<p align="center">
  <img src="githubDecors/example.gif" width="600" alt="GUI Demo"/>
  <br/>
  <i>Example: Interactive GUI for watermarking and steganography</i>
</p>

---

## ✨ Features

| Feature      | Support |
|--------------|:-------:|
| DCT          |   ✅    |
| DFT          |   ✅    |
| DWT          |   ✅    |
| LSB          |   ✅    |
| DSSS         |   ✅    |
| Text Watermarking | ✅ |
| Image Watermarking | ✅ |
| GUI          |   ✅    |

---

## 📋 Prerequisites

### System Requirements
- **Python 3.13.3 or newer**
- **Tkinter 8.6 or newer** (mandatory, for GUI)
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### Operating System Support
- ✅ **Windows 10/11**
- ✅ **Linux** (Ubuntu 20.04+, Debian 11+, etc.)
- ✅ **macOS** (10.15+)

### Python Installation

#### Windows
1. **Download Python from [python.org](https://www.python.org/downloads/)**
2. **During installation, make sure to check:**
   - ✅ "Add Python to PATH"
   - ✅ "Install pip"
   - ✅ "Install Tkinter" (usually included by default)

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-tk git
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python3

# Or download from python.org
# Make sure Tkinter is included
```

### Verify Installation
```bash
# Check Python version
python --version  # Should be 3.13.3 or newer

# Check pip
pip --version

# Check Tkinter
python -c "import tkinter; print('Tkinter is available')"
```

---

## 📦 Getting Started

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/DarkPhoenixz/MultimediaProject.git
   cd MultimediaProject
   ```

2. **Run the boot script:**
   ```sh
   python boot_script.py
   ```
   
   The boot script will:
   - ✅ Check Python and Tkinter versions
   - ✅ Set up a virtual environment
   - ✅ Install all dependencies from `requirements.txt`
   - ✅ Prepare the project for first use
   - 🚀 Launch the GUI automatically

   **After installation, the GUI will launch automatically.**

#### Boot Script Flags
- `--no-emoji` : Disables emoji in log and console output
- `--force-recreate` : Forces recreation of the virtual environment

Example:
```sh
python boot_script.py --no-emoji --force-recreate
```

#### Manual Installation (Alternative)
If the boot script fails, you can install manually:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## 🖥️ Usage

### 1. **Using the GUI (Recommended)**
- The GUI opens automatically after installation, or run:
  ```sh
  python app.py
  ```
- **Select the images** for cover and watermark/secret from the `images/` folder.
- Choose the algorithm, load images, and visualize results interactively.
- **Strongly recommended** for best experience and to avoid manual errors.

### 2. **Running Algorithms from Terminal (Advanced/Optional)**
- Example (DCT full image watermarking):
  ```sh
  python algorithms/DCT/Image/DCT_full.py images/lena.png images/mark.png
  ```
- Use images from the `images/` folder.
- Output images and metrics will be saved in the working directory.
- **Note:** Direct script execution is possible but the GUI is preferred.

---

## 🧪 Testing

### Running Tests
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Run all tests
pytest

# Run specific test file
pytest tests/test_dct.py

# Run with verbose output
pytest -v
```

### Test Notes
- Tests use real images from the `images/` folder
- SSIM (Structural Similarity Index) is used for visual quality assessment
- Some tests may be marked as "expected fail" (xfail) due to algorithm limitations
- Visual evaluation is recommended alongside automated tests

---

## 📝 Notes
- `.gitignore` excludes macOS system files and unnecessary artifacts.
- For any issues, use `boot_script.py` for guided setup.
- The project uses a virtual environment to avoid conflicts with system Python packages.

---

## 👤 Authors
- Matteo Gallina <matt.gallina@gmail.com>
- Graziana Calderaro <your.email@example.com>
- Emily Gigliuto <your.email@example.com>
