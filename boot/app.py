import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import subprocess
import os
import sys
from pathlib import Path

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography & Watermarking")
        self.root.geometry("900x650")
        self.root.configure(bg="#f8f9fa")

        self.selected_algorithm_name = tk.StringVar(value="Choose algorithm")
        self.selected_algorithm_path = None

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", background="#f8f9fa", font=("Segoe UI", 11))

        title = tk.Label(self.root, text="Steganography & Watermarking Tool", font=("Segoe UI", 16, "bold"), bg="#f8f9fa")
        title.pack(pady=10)

        # Frame immagini
        self.frame = tk.Frame(self.root, bg="#f8f9fa")
        self.frame.pack(pady=10)

        self.frameV = tk.Frame(self.frame, bg="#f8f9fa")
        self.frameV.pack(side="left", padx=30)

        self.frameV2 = tk.Frame(self.frame, bg="#f8f9fa")
        self.frameV2.pack(side="left", padx=30)

        self.canvas = tk.Canvas(self.frameV, width=256, height=256, bg='#dee2e6', bd=2, relief="ridge")
        self.canvas.pack()
        self.upload_button = ttk.Button(self.frameV, text="Upload Image",
                                        command=lambda: self.load_image(self.canvas, "image_path"))
        self.upload_button.pack(pady=10)

        self.canvas2 = tk.Canvas(self.frameV2, width=256, height=256, bg='#dee2e6', bd=2, relief="ridge")
        self.canvas2.pack()
        self.upload_button2 = ttk.Button(self.frameV2, text="Upload Logo",
                                         command=lambda: self.load_image(self.canvas2, "logo_path"))
        self.upload_button2.pack(pady=10)

        self.algorithm_button = ttk.Button(self.root, textvariable=self.selected_algorithm_name, command=self.show_algorithm_menu)
        self.algorithm_button.pack(pady=15)

        self.text_label = ttk.Label(self.root, text="Insert stego text:")
        self.text_label.pack()
        self.text_label.pack_forget()

        self.text_input = tk.Text(self.root, width=60, height=4, font=("Segoe UI", 10))
        self.text_input.pack(pady=5)
        self.text_input.pack_forget()

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_algorithm)
        self.start_button.pack(pady=20)

        self.build_algorithm_menu()

    def build_algorithm_menu(self):
        self.algo_menu = tk.Menu(self.root, tearoff=0)
        base_path = Path(__file__).parent.parent / "algorithms"

        if not base_path.exists():
            messagebox.showerror("Error", "The 'algorithms' folder was not found.")
            return

        def build_menu_recursive(parent_menu, current_path):
            for item in sorted(current_path.iterdir()):
                if item.is_dir() and not item.name.startswith("__"):
                    submenu = tk.Menu(parent_menu, tearoff=0)
                    build_menu_recursive(submenu, item)
                    parent_menu.add_cascade(label=item.name, menu=submenu)
                elif item.suffix == ".py" and not item.name.startswith("__"):
                    label = item.stem
                    relative_path = item.relative_to(base_path)
                    parent_menu.add_command(
                        label=label,
                        command=lambda p=str(relative_path): self.set_algorithm(p)
                    )

        build_menu_recursive(self.algo_menu, base_path)

    def show_algorithm_menu(self):
        x = self.algorithm_button.winfo_rootx()
        y = self.algorithm_button.winfo_rooty() + self.algorithm_button.winfo_height()
        self.algo_menu.tk_popup(x, y)

    def set_algorithm(self, algo_relative_path):
        self.selected_algorithm_path = algo_relative_path
        algo_name = Path(algo_relative_path).stem
        self.selected_algorithm_name.set(algo_name)

        if algo_name.lower() == "dct_text":
            self.canvas2.pack_forget()
            self.upload_button2.pack_forget()
            self.text_label.pack(pady=5)
            self.text_input.pack(pady=5)
        else:
            if not self.canvas2.winfo_ismapped():
                self.canvas2.pack()
                self.upload_button2.pack(pady=10)
            self.text_input.pack_forget()
            self.text_label.pack_forget()

    def load_image(self, canvas, attr_name):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.bmp")])
        if file_path:
            img = Image.open(file_path).convert("RGB")
            img.thumbnail((256, 256))
            photo = ImageTk.PhotoImage(img)
            canvas.image = photo
            canvas.delete("all")
            canvas.create_image(128, 128, image=photo)
            setattr(self, attr_name, file_path)

    def start_algorithm(self):
        if not self.selected_algorithm_path:
            messagebox.showwarning("Warning", "Please select an algorithm before starting.")
            return

        if not hasattr(self, 'image_path'):
            messagebox.showwarning("Warning", "Please upload an image before starting.")
            return

        algo_name = self.selected_algorithm_name.get().lower()
        text_input = None

        if algo_name == "dct_text":
            text_input = self.text_input.get("1.0", tk.END).strip()
            if not text_input:
                messagebox.showwarning("Warning", "Please enter the text to embed.")
                return
        else:
            if not hasattr(self, 'logo_path'):
                messagebox.showwarning("Warning", "Please upload a logo before starting.")
                return

        project_root = os.path.dirname(os.path.dirname(__file__))
        script_path = os.path.join(project_root, "algorithms", self.selected_algorithm_path)
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script '{self.selected_algorithm_path}' not found.")
            return

        args = [sys.executable, script_path, self.image_path]
        args.append(text_input if text_input else self.logo_path)

        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Script execution failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
