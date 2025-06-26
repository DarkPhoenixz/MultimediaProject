import numpy as np
from PIL import Image

def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Calcola SSIM (Structural Similarity Index) tra due immagini.
    
    Args:
        img1, img2: array numpy delle immagini (stessa dimensione)
        window_size: dimensione della finestra per il calcolo locale
        sigma: deviazione standard del filtro gaussiano
    
    Returns:
        float: valore SSIM tra -1 e 1 (1 = identiche)
    """
    # Assicurati che le immagini siano float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Parametri SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calcola media locale
    mu1 = _gaussian_filter(img1, sigma, window_size)
    mu2 = _gaussian_filter(img2, sigma, window_size)
    
    # Calcola varianza locale
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _gaussian_filter(img1 ** 2, sigma, window_size) - mu1_sq
    sigma2_sq = _gaussian_filter(img2 ** 2, sigma, window_size) - mu2_sq
    sigma12 = _gaussian_filter(img1 * img2, sigma, window_size) - mu1_mu2
    
    # Calcola SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def _gaussian_filter(img, sigma, window_size):
    """Filtro gaussiano semplificato per SSIM."""
    # Implementazione semplificata del filtro gaussiano
    # In pratica, usiamo una media mobile
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=window_size)

def compute_mse(img1, img2):
    """Calcola MSE tra due immagini."""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def compute_psnr(mse, max_pixel=255.0):
    """Calcola PSNR dato MSE."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def load_test_images():
    """Carica le immagini di test standard."""
    import os
    from PIL import Image
    
    cover_path = "images/lena.png"
    watermark_path = "images/mark.png"
    
    if not os.path.exists(cover_path) or not os.path.exists(watermark_path):
        return None, None
    
    cover_img = Image.open(cover_path).convert('L')
    watermark_img = Image.open(watermark_path).convert('L')
    
    # Ridimensiona il watermark se necessario
    if watermark_img.size != cover_img.size:
        watermark_img = watermark_img.resize(cover_img.size)
    
    return np.array(cover_img), np.array(watermark_img) 