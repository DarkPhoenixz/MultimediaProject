# MultimediaProject

## Overview

MultimediaProject is a comprehensive toolkit for image steganography and watermarking, featuring multiple algorithms (DCT, DFT, DWT, LSB, DSSS) for both image and text data. The project is designed for research, experimentation, and educational purposes, providing modular scripts and a GUI for easy interaction.

## Features
- Multiple watermarking and steganography algorithms (DCT, DFT, DWT, LSB, DSSS)
- Support for both image and text watermarking
- Modular Python scripts for each algorithm
- Example images and results included
- Cross-platform support (Windows, Linux, macOS)
- Professional bootstrapping and environment setup

## Getting Started

### Prerequisites
- **Python 3.8+** (Recommended: Python 3.10 or newer)
- [pip](https://pip.pypa.io/en/stable/)
- [Git](https://git-scm.com/)
- (Optional) [Tkinter](https://wiki.python.org/moin/TkInter) for GUI support (usually included with Python)

### Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/MultimediaProject.git
   cd MultimediaProject
   ```
2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### 1. **Run the GUI Application**
   ```sh
   python boot_script.py
   ```
   - The GUI allows you to select algorithms, load images, and visualize results interactively.

#### 2. **Run Individual Algorithms**
   Each algorithm is available as a standalone script. Example (DCT full image watermarking):
   ```sh
   python algorithms/DCT/Image/DCT-full.py images/lena.png images/mark.png
   ```
   - Replace the image paths as needed.
   - Output images and metrics will be saved in the working directory.

#### 3. **Batch Run All Algorithms**
   ```sh
   python runAll.py
   ```

### Testing

- **Manual Testing:**
  - Run each script in the `algorithms/` directory with sample images provided in the `images/` folder.
  - Check the output images and console metrics (MSE, PSNR, etc.).

- **Automated Testing:**
  - (If available) Run test scripts or use `pytest` for any test modules.
  - Example:
    ```sh
    pytest
    ```

### Notes
- The `.gitignore` is configured to exclude macOS system files and other unnecessary artifacts.
- For any issues with dependencies or environment, use `boot_script.py` for guided setup.

## License
[MIT License](LICENSE)

## Authors
- Your Name <your.email@example.com>
- Contributors: See GitHub commit history