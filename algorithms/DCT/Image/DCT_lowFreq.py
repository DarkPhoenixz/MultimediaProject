#grazy Watermarking
import timeit
import numpy as np
from PIL import Image
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import sys

def embed_watermark_dct(cover_image_path, watermark_image_path, stego_image_path, alpha=10):
    # Load the cover image and the watermark (resized to 128x128)
    cover_img = Image.open(cover_image_path).convert('L')
    watermark_img = Image.open(watermark_image_path).convert('L').resize((128, 128))
    
    cover_np = np.array(cover_img, dtype=np.float32)
    watermark_np = np.array(watermark_img, dtype=np.float32)
    
    # Normalize the watermark to [0,1]
    watermark_norm = watermark_np / 255.0
    
    # Calculate the DCT of the cover image
    cover_dct = fftpack.dct(fftpack.dct(cover_np.T, norm='ortho').T, norm='ortho')
    
    # Embed the watermark by modifying DCT coefficients (top-left corner)
    cover_dct[:128, :128] += alpha * watermark_norm
    
    # Calculate the inverse DCT to obtain the stego image
    stego_np = fftpack.idct(fftpack.idct(cover_dct.T, norm='ortho').T, norm='ortho')
    stego_np_clipped = np.clip(stego_np, 0, 255).astype(np.uint8)
    
    stego_img = Image.fromarray(stego_np_clipped)
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path}")
    
    return cover_np, watermark_np, stego_np_clipped

def extract_watermark_dct(stego_image_path, cover_image_path, alpha=10, watermark_size=(128,128)):
    # Load the stego and cover images
    stego_img = Image.open(stego_image_path).convert('L')
    cover_img = Image.open(cover_image_path).convert('L')
    
    stego_np = np.array(stego_img, dtype=np.float32)
    cover_np = np.array(cover_img, dtype=np.float32)
    
    # Calculate the DCT for both images
    stego_dct = fftpack.dct(fftpack.dct(stego_np.T, norm='ortho').T, norm='ortho')
    cover_dct = fftpack.dct(fftpack.dct(cover_np.T, norm='ortho').T, norm='ortho')
    
    # Extract the watermark from the difference of DCT coefficients
    extracted_watermark = (stego_dct[:watermark_size[0], :watermark_size[1]] - 
                             cover_dct[:watermark_size[0], :watermark_size[1]]) / alpha
    
    # Scale the extracted watermark to 0-255
    extracted_watermark = np.clip(extracted_watermark * 255, 0, 255).astype(np.uint8)
    
    return extracted_watermark

def compute_mse(imageA, imageB):
    """Calculates Mean Squared Error between two images."""
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def compute_psnr(mse, max_pixel=255.0):
    """Calculates PSNR (Peak Signal-to-Noise Ratio) given MSE."""
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

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

def main(cover_image_path, secret_image_path):
    stego_image_path = 'stego_dct.png'
    alpha = 7
    cover_np, watermark_np, stego_np = embed_watermark_dct(cover_image_path, secret_image_path, stego_image_path, alpha=alpha)
    extracted_watermark = extract_watermark_dct(stego_image_path, cover_image_path, alpha=alpha)
    embedTime = timeit.timeit(
        lambda: embed_watermark_dct(cover_image_path, secret_image_path, stego_image_path, alpha=alpha),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_watermark_dct(stego_image_path, cover_image_path, alpha=alpha),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, stego_np)
    mse_watermark = compute_mse(watermark_np.astype(np.uint8), extracted_watermark)
    psnr_watermark = compute_psnr(mse_watermark)
    ssim_watermark = compute_ssim(watermark_np.astype(np.uint8), extracted_watermark)
    print(f"Cover vs. Watermark: MSE = {mse_cover:.2f}, PSNR = {psnr_cover:.2f} dB")
    print(f"Watermark vs. Extracted: MSE = {mse_watermark:.2f}, PSNR = {psnr_watermark:.2f} dB")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(cover_np, cmap='gray')
    axes[0,0].set_title("Cover Image")
    axes[0,0].axis("off")
    axes[0,1].imshow(stego_np, cmap='gray')
    axes[0,1].set_title("Watermark Image")
    axes[0,1].axis("off")
    axes[1,0].imshow(watermark_np.astype(np.uint8), cmap='gray')
    axes[1,0].set_title("Original Watermark")
    axes[1,0].axis("off")
    axes[1,1].imshow(extracted_watermark, cmap='gray')
    axes[1,1].set_title("Extracted Watermark")
    axes[1,1].axis("off")
    fig.suptitle(
        f"Cover vs. Watermark MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB, SSIM: {ssim_cover:.4f}\n"
        f"Watermark vs. Extracted MSE: {mse_watermark:.2f}, PSNR: {psnr_watermark:.2f} dB, SSIM: {ssim_watermark:.4f}\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DCT_lowFreq.py original_image secret_image")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]
    print("Received:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#piu robusto alle compressioni 
#i wm con linee nette sono complessi da incorporare, volendo si potrebbe sfocare
#honorable mention: lena con dentro lena
    