import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image_grayscale(image_path):
    """Carica un'immagine e la converte in scala di grigi."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Errore: impossibile caricare l'immagine {image_path}")
    
    return image

def apply_dft(image):
    """Calcola la DFT e sposta lo zero-frequenza al centro."""
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def inverse_dft(dft_shifted):
    """Applica l'inversa della DFT e ritorna l'immagine ricostruita."""
    f_ishift = np.fft.ifftshift(dft_shifted)
    image_back = np.fft.ifft2(f_ishift)
    return np.abs(image_back)

def embed_watermark(dft_shifted, watermark, alpha):
    """Incorpora il watermark nella trasformata di Fourier."""
    magnitude = np.abs(dft_shifted)
    phase = np.angle(dft_shifted)
    
    # Normalizza il watermark per evitare distorsioni
    mean_val = np.mean(watermark)  # Trova il valore medio
    watermark = watermark - mean_val  # Rimuove la componente DC
    watermark = watermark / np.max(np.abs(watermark))  # Normalizza tra -1 e 1

    # Somma il watermark alla magnitudine
    watermarked_magnitude = magnitude * (1 + alpha * watermark)

    # Ricostruisci la DFT con la magnitudine modificata
    watermarked_dft = watermarked_magnitude * np.exp(1j * phase)
    return watermarked_dft

def resize_watermark(watermark, target_shape):
    """Ridimensiona il watermark per adattarlo all'immagine target."""
    return cv2.resize(watermark, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

def display_images(original, watermarked):
    """Mostra le immagini originali e watermarked in due finestre."""
    cv2.imshow("Original Image", original)
    cv2.imshow("Watermarked Image", watermarked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_magnitude_spectrum(dft_image, title="Magnitude Spectrum"):
    """Mostra lo spettro di magnitudine della DFT"""
    magnitude_spectrum = 20 * np.log(np.abs(dft_image) + 1)
    plt.figure()
    plt.title(title)
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.colorbar()
    plt.show()

def calculate_psnr(original, watermarked):
    """Calcola il PSNR tra due immagini"""
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10(255**2 / mse)
    return psnr

def extract_watermark(original_dft, watermarked_dft, alpha):
    """Estrae il watermark dalla DFT dell'immagine watermarked"""
    magnitude_original = np.abs(original_dft)
    magnitude_watermarked = np.abs(watermarked_dft)
    extracted_watermark = (magnitude_watermarked - magnitude_original) / alpha
    return extracted_watermark

def main():
    # Carica l'immagine principale (Lena)
    lena_image = load_image_grayscale("../../lena.png")
    
    # Carica il watermark (Logo)
    logo_image = load_image_grayscale("../../mark.png")
    
    # Ridimensiona il watermark per adattarlo all'immagine principale
    logo_resized = resize_watermark(logo_image, lena_image.shape)
    
    # Applica la DFT all'immagine principale
    dft_shifted = apply_dft(lena_image)
    show_magnitude_spectrum(dft_shifted, "Original DFT Spectrum")
    
    # Incorpora il watermark
    alpha = 0.01  # Intensità del watermark
    watermarked_dft = embed_watermark(dft_shifted, logo_resized, alpha)
    show_magnitude_spectrum(watermarked_dft, "Watermarked DFT Spectrum")
    
    # Recupera l'immagine con l'inversa della DFT
    watermarked_image = inverse_dft(watermarked_dft)
    
    # Normalizza l'immagine per visualizzazione corretta
    watermarked_image = cv2.convertScaleAbs(watermarked_image, alpha=(255.0 / np.max(watermarked_image)))

    
    # Mostra le immagini
    display_images(lena_image, watermarked_image)
    
    # Calcola il PSNR
    psnr_value = calculate_psnr(lena_image, watermarked_image)
    print(f"PSNR tra immagine originale e watermarked: {psnr_value:.2f} dB")
    
    # Estrai il watermark
    extracted_watermark = extract_watermark(dft_shifted, watermarked_dft, alpha)
    cv2.imshow("Extracted Watermark", extracted_watermark)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Salva l'immagine risultante
    cv2.imwrite("watermarked_lena.png", watermarked_image)

if __name__ == "__main__":
    main()
