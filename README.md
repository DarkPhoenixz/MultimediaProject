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
- **Python 3.13.3+** (Required)
- **Tkinter 8.6+** (Required for GUI; must be installed and available in your Python environment)
- [pip](https://pip.pypa.io/en/stable/)
- [Git](https://git-scm.com/)

### Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/MultimediaProject.git
   cd MultimediaProject
   ```
2. **Run the boot script:**
   ```sh
   python boot_script.py
   ```
   The boot script will automatically:
   - Check your Python and Tkinter versions
   - Set up a virtual environment if needed
   - Install all required dependencies
   - Prepare the project for first use

   **No manual environment or dependency setup is required!**

   **After installation, the GUI will launch automatically.**

#### Boot Script Flags
You can use the following flags when launching the boot script:
- `--no-emoji` : Disables emoji in log and console output (useful for terminals that do not support Unicode/emoji).
- `--force-recreate` : Forces the recreation of the virtual environment, even if one already exists (useful if your environment is broken or you want a clean setup).

Example:
```sh
python boot_script.py --no-emoji --force-recreate
```

### Usage

#### 1. **Using the GUI Application (Recommended)**
   - After installation, the GUI will open automatically. If not, you can launch it with:
     ```sh
     python boot_script.py
     ```
   - **Select the images you want to use** for cover and watermark/secret from the `images/` folder included in the project.
   - The GUI allows you to choose the algorithm, load images, and visualize results interactively.
   - **It is strongly recommended to use the GUI** for the best experience and to avoid manual errors.

#### 2. **Running Individual Algorithms from Terminal (Advanced/Optional)**
   Each algorithm is available as a standalone script. Example (DCT full image watermarking):
   ```sh
   python algorithms/DCT/Image/DCT-full.py images/lena.png images/mark.png
   ```
   - Replace the image paths as needed (use images from the `images/` folder).
   - Output images and metrics will be saved in the working directory.
   - **Note:** Direct script execution is possible but the GUI is preferred for ease of use.

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