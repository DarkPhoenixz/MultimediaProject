import subprocess
import re
import json
import os

# Percorsi degli script da eseguire
algorithms = {
    "DCT": [
        "DCT/Image/DCT-full.py",
        "DCT/Image/DCT-JPEG-like.py",
        "DCT/Image/DCT-lowFreq.py",
        "DCT/Image/DCT-singleQuant.py",
        "DCT/Text/DCT-text.py"
    ],
    "DFT": [
        "DFT/Image/DFT.py"
    ],
    "DSSS": [
        "DSSS/Image/DSSS.py"
    ],
    "DWT": [
        "DWT/Image/DWT-multiLevel.py",
        "DWT/Image/DWT-oneLevel.py",
        "DWT/Image/DWT-otsu.py"
    ],
    "LSB": [
        "LSB/Image/LSB-spatial.py"
    ]
}

# Output file
output_file = "results.json"

# Dizionario per i risultati
results = {}

# Esegui ogni script nel suo contesto corretto
for category, scripts in algorithms.items():
    results[category] = {}

    for script in scripts:
        algo_name = os.path.basename(script).replace(".py", "")
        script_dir = os.path.dirname(script)

        print(f"Eseguendo {algo_name} in {script_dir}...")

        process = subprocess.Popen(["python", os.path.basename(script)],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   cwd=script_dir,
                                   creationflags=subprocess.CREATE_NO_WINDOW)

        stdout, stderr = process.communicate()

        if stderr:
            print(f"⚠️ Errore in {algo_name}: {stderr}")
            results[category][algo_name] = {"error": stderr}
            continue

        mse_cover = re.findall(r"Cover vs\. Watermarked MSE:\s?([\d\.e\-]+)", stdout)
        psnr_cover = re.findall(r"Cover vs\. Watermarked.*?PSNR:\s?([\d\.]+)", stdout)
        mse_watermark = re.findall(r"Watermark vs\. Extracted MSE:\s?([\d\.e\-]+)", stdout)
        psnr_watermark = re.findall(r"Watermark vs\. Extracted.*?PSNR:\s?([\d\.]+)", stdout)
        embed_time = re.findall(r"(?i)Embedding Time:\s?([\d\.]+)", stdout)
        extraction_time = re.findall(r"(?i)Extraction Time:\s?([\d\.]+)", stdout)

        results[category][algo_name] = {
            "mse_cover": float(mse_cover[0]) if mse_cover else None,
            "psnr_cover": float(psnr_cover[0]) if psnr_cover else None,
            "mse_watermark": float(mse_watermark[0]) if mse_watermark else None,
            "psnr_watermark": float(psnr_watermark[0]) if psnr_watermark else None,
            "embedding_time": float(embed_time[0]) if embed_time else None,
            "extraction_time": float(extraction_time[0]) if extraction_time else None
        }

with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)

print("\n✅ Tutti gli algoritmi sono stati eseguiti. Risultati salvati in 'results.json'.")