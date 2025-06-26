import timeit
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys



def generate_pseudo_random_sequence(seed, shape):
    """Genera una sequenza pseudo-casuale con distribuzione normale."""
    np.random.seed(seed)
    return np.random.randn(*shape)

def mse(image1, image2):
    """Calcola l'errore quadratico medio (MSE)."""
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    """Calcola il rapporto segnale-rumore di picco (PSNR)."""
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse_value)

def embed_watermark(image_path, watermark_path, alpha, seed=42):
    """Incorpora il watermark nell'immagine utilizzando DSS."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    if image is None or watermark is None:
        raise FileNotFoundError("Errore: Immagine o watermark non trovati!")

    image = image.astype(np.float32)
    watermark = watermark.astype(np.float32)
    # Ridimensiona il watermark per adattarlo all'immagine
    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))
    # Converti in binario: pixel >127 diventano 1, altrimenti 0
    _, watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)

    # Applica la DCT all'immagine
    dct_image = cv2.dct(image)
    # Genera sequenza pseudo-casuale (rumore)
    noise = generate_pseudo_random_sequence(seed, dct_image.shape)
    # Incorpora il watermark nella DCT
    dct_watermarked = dct_image + alpha * watermark * noise
    # Ricostruisce l'immagine watermarked tramite IDCT
    watermarked_image = cv2.idct(dct_watermarked)
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8), watermark, watermarked_image

def extract_watermark(watermarked, original, alpha, seed=42):
    """Estrae il watermark utilizzando DSS."""
    watermarked = watermarked.astype(np.float32)
    original = original.astype(np.float32)
    dct_watermarked = cv2.dct(watermarked)
    dct_original = cv2.dct(original)
    noise = generate_pseudo_random_sequence(seed, dct_original.shape)
    extracted_watermark = (dct_watermarked - dct_original) / (alpha * noise)
    extracted_watermark = np.clip(extracted_watermark * 255, 0, 255).astype(np.uint8)
    return extracted_watermark

def display_combined_results(original, watermark, watermarked, extracted_watermark, embedTime, extractTime):
    """Mostra tutti i risultati in un'unica finestra matplotlib."""
    # Calcola gli spettri DCT per visualizzazione
    dct_original = cv2.dct(original.astype(np.float32))
    dct_watermarked = cv2.dct(watermarked.astype(np.float32))
    spectrum_original = 20 * np.log(np.abs(dct_original) + 1)
    spectrum_watermarked = 20 * np.log(np.abs(dct_watermarked) + 1)
    
    # Calcola metriche per l'immagine principale
    mse_main = mse(original, watermarked)
    psnr_main = psnr(original, watermarked)
    
    # Calcola metriche per il watermark
    # Normalizza il watermark originale (binario: 0 o 1) e l'estratto (convertito a [0,1])
    wm_orig = watermark.astype(np.float32)
    wm_extr = extracted_watermark.astype(np.float32) / 255.0
    mse_wm = mse(wm_orig, wm_extr)
    psnr_wm = 10 * np.log10(1.0 / mse_wm) if mse_wm != 0 else float('inf')
    
    # Crea figura con 2 righe e 3 colonne
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Immagini
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Originale")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(watermark * 255, cmap="gray")  # Moltiplica per 255 per visualizzare il binario
    axes[0, 1].set_title("Watermark (binario)")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(watermarked, cmap="gray")
    axes[0, 2].set_title("Watermarked")
    axes[0, 2].axis("off")
    
    # Row 2: Spettri e watermark estratto
    im1 = axes[1, 0].imshow(spectrum_original, cmap="gray")
    axes[1, 0].set_title("Spettro DCT Originale")
    axes[1, 0].axis("off")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(spectrum_watermarked, cmap="gray")
    axes[1, 1].set_title("Spettro DCT Watermarked")
    axes[1, 1].axis("off")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(extracted_watermark, cmap="gray")
    axes[1, 2].set_title("Watermark Estratto")
    axes[1, 2].axis("off")
    
    # Aggiungi le metriche come suptitle
    fig.suptitle(f"Main Image - PSNR: {psnr_main:.2f} dB, MSE: {mse_main:.2f} | "
                 f"Watermark - PSNR: {psnr_wm:.2f} dB, MSE: {mse_wm:.2e}\n"
                 f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
                   fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main(cover_image_path, secret_image_path):
    
    # Incorpora il watermark
    original, watermark, watermarked = embed_watermark(cover_image_path, secret_image_path, alpha=1, seed=42)
    # Estrae il watermark
    extracted_watermark = extract_watermark(watermarked, original, alpha=1, seed=42)
    
    # Calculate embedding computational time
    def embed_wrapper():
        embed_watermark(cover_image_path, secret_image_path, alpha=1, seed=42)
    embedTime = timeit.timeit(embed_wrapper, number=10)     

    # Calculate extraction computational time
    def extract_wrapper():
        extract_watermark(watermarked, original, alpha=1, seed=42)
    extractTime = timeit.timeit(extract_wrapper, number=10)
    
    # Mostra tutti i risultati in una sola finestra
    display_combined_results(original, watermark, watermarked, extracted_watermark, embedTime, extractTime)
    
    # Salva l'immagine watermarked ed eventualmente il watermark estratto
    cv2.imwrite("watermarked_dss.png", watermarked)
    cv2.imwrite("extracted_watermark_dss.png", extracted_watermark)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python DSSS.py original_image secret_image")
        sys.exit(1)

    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]

    print("Ricevuto:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)


#il risultato è un po scadente questo a causa delle proprietà del dsss
#si cerca robustezza e segretezza
#cmq è leggibile

#si potrebbe pensare ad un decoding piu avanzato che includa denoising 
#ALZARE ALPHAAAA