#grazy Watermarking
import timeit
import sys
import numpy as np
from PIL import Image
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

def pad_watermark_to_cover(watermark_np, target_shape):
    """Pads the watermark to match the target shape (centered)."""
    h_wm, w_wm = watermark_np.shape
    h_tgt, w_tgt = target_shape

    if h_wm > h_tgt or w_wm > w_tgt:
        raise ValueError("Watermark is larger than cover image. Cannot pad.")

    pad_top = (h_tgt - h_wm) // 2
    pad_bottom = h_tgt - h_wm - pad_top
    pad_left = (w_tgt - w_wm) // 2
    pad_right = w_tgt - w_wm - pad_left

    padded = np.pad(watermark_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return padded

def embed_watermark_dct(cover_image, secret_image, alpha):

    cover_np = np.array(cover_image, dtype=np.float32)
    watermark_np = np.array(secret_image, dtype=np.float32)

    # Normalizza il watermark in [0, 1]
    watermark_norm = watermark_np / 255.0

    # Calcola la DCT bidimensionale dell'immagine di copertura
    cover_dct = fftpack.dct(fftpack.dct(cover_np.T, norm='ortho').T, norm='ortho')

    # Incorpora il watermark nelle componenti DCT
    watermarked_dct = cover_dct + alpha * watermark_norm

    # Calcola la DCT inversa per ottenere l'immagine watermarked
    watermarked_np = fftpack.idct(fftpack.idct(watermarked_dct.T, norm='ortho').T, norm='ortho')
    watermarked_np_clipped = np.clip(watermarked_np, 0, 255).astype(np.uint8)

    watermarked_img = Image.fromarray(watermarked_np_clipped)

    return cover_np, watermark_np, watermarked_np_clipped, watermarked_img

def extract_watermark_dct(watermarked_img, cover_img, alpha):
    
    watermarked_np = np.array(watermarked_img, dtype=np.float32)
    cover_np = np.array(cover_img, dtype=np.float32)

    # Calcola la DCT per entrambe le immagini
    watermarked_dct = fftpack.dct(fftpack.dct(watermarked_np.T, norm='ortho').T, norm='ortho')
    cover_dct = fftpack.dct(fftpack.dct(cover_np.T, norm='ortho').T, norm='ortho')
    
    # Estrai il watermark calcolando la differenza fra i coefficienti
    extracted_watermark = (watermarked_dct - cover_dct) / alpha
    
    # Riporta il watermark estratto nella scala 0-255
    extracted_watermark = np.clip(extracted_watermark * 255, 0, 255).astype(np.uint8)
    
    return extracted_watermark

def compute_mse(imageA, imageB):
    """Calcola il Mean Squared Error (MSE) tra due immagini."""
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def compute_psnr(mse, max_pixel=255.0):
    """Calcola il PSNR (Peak Signal-to-Noise Ratio) dato l'MSE."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = _gaussian_filter(img1, sigma, window_size)
    mu2 = _gaussian_filter(img2, sigma, window_size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _gaussian_filter(img1 ** 2, sigma, window_size) - mu1_sq
    sigma2_sq = _gaussian_filter(img2 ** 2, sigma, window_size) - mu2_sq
    sigma12 = _gaussian_filter(img1 * img2, sigma, window_size) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def _gaussian_filter(img, sigma, window_size):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma, truncate=((window_size-1)/2)/sigma)

def main(cover_image_path, secret_image):
    watermarked_image_path = 'watermark_dct_full.png'
    cover_img = Image.open(cover_image_path).convert('L')
    secret_img = Image.open(secret_image).convert('L')
    if secret_img.size > cover_img.size:
        secret_img = secret_img.resize(cover_img.size)
    elif secret_img.size < cover_img.size:
        secret_image = pad_watermark_to_cover(np.array(secret_img), np.array(cover_img).shape)
        secret_img = Image.fromarray(secret_image)
    alpha = 1
    cover_np, watermark_np, watermarked_np, watermarked_img = embed_watermark_dct(cover_img, secret_img, alpha)
    watermarked_img.save(watermarked_image_path)
    print(f"Watermarked image saved to {watermarked_image_path}")
    watermarked_img = Image.open(watermarked_image_path).convert('L')
    extracted_watermark = extract_watermark_dct(watermarked_img, cover_img, alpha=alpha)
    embedTime = timeit.timeit(
        lambda: embed_watermark_dct(cover_img, secret_img, alpha),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_watermark_dct(watermarked_img, cover_img, alpha=alpha),
        number=10
    )
    mse_cover = compute_mse(cover_np, watermarked_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, watermarked_np)
    mse_watermark = compute_mse(watermark_np.astype(np.uint8), extracted_watermark)
    psnr_watermark = compute_psnr(mse_watermark)
    ssim_watermark = compute_ssim(watermark_np.astype(np.uint8), extracted_watermark)
    print(f"Cover vs. Watermarked: MSE = {mse_cover:.2f}, PSNR = {psnr_cover:.2f} dB")
    print(f"Watermark vs. Extracted: MSE = {mse_watermark:.2f}, PSNR = {psnr_watermark:.2f} dB")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(cover_np, cmap='gray')
    axes[0,0].set_title("Cover Image")
    axes[0,0].axis("off")
    axes[0,1].imshow(watermarked_np, cmap='gray')
    axes[0,1].set_title("Watermarked Image")
    axes[0,1].axis("off")
    axes[1,0].imshow(watermark_np.astype(np.uint8), cmap='gray')
    axes[1,0].set_title("Original Watermark")
    axes[1,0].axis("off")
    axes[1,1].imshow(extracted_watermark, cmap='gray')
    axes[1,1].set_title("Extracted Watermark")
    axes[1,1].axis("off")
    fig.suptitle(
        f"Cover vs. Watermarked MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB, SSIM: {ssim_cover:.4f}\n"
        f"Watermark vs. Extracted MSE: {mse_watermark:.2f}, PSNR: {psnr_watermark:.2f} dB, SSIM: {ssim_watermark:.4f}\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python DCT_full.py original_image secret_image")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_image = sys.argv[2]
    print("Received:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image)
    main(cover_image_path, secret_image)

#dovrebbe essere ok, con lena funziona bene. piu è semplice il watermark meglio è
#da cosa è dovuto?
