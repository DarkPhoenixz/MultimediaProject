#Matteo Steganografia 
import timeit
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys



# For Otsu threshold & morphological operations
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, closing


def quantize_to_n_bits(image_array, n):
    """
    Quantize an 8-bit grayscale image to n bits.
    For n=1, the image will have 2^1 = 2 gray levels (0 or 1 in quantized form).
    """
    levels = 2 ** n
    quantized = (image_array.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)
    return quantized

def dequantize_from_n_bits(quantized_array, n):
    """
    Scale a quantized image (n bits) back to the 8-bit range.
    For n=1, maps 0->0 and 1->255.
    """
    levels = 2 ** n
    dequantized = (quantized_array.astype(np.uint16) * 255 // (levels - 1)).astype(np.uint8)
    return dequantized

def compute_mse(imgA, imgB):
    return np.mean((imgA.astype(float) - imgB.astype(float)) ** 2)

def compute_psnr(mse, max_val=255.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

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
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=window_size)


def embed_secret_image_dwt_distributed(
    cover_image_path,
    secret_image_path,
    stego_image_path,
    bits_used=1,            # Use 1 bit for embedding
    wavelet='bior2.2',      # Biorthogonal wavelet
    level=1                 # Single-level DWT
):
    """
    Embed a secret image into a cover image by distributing the secret data
    over the detail subbands of a multi-level (here single-level) DWT.
    """
    # Load cover
    cover_img = Image.open(cover_image_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.uint8)
    
    # Decompose cover
    coeffs = pywt.wavedec2(cover_np, wavelet=wavelet, level=level)
    # coeffs[0] = cA, coeffs[1] = (cH, cV, cD) for level=1
    
    # Calculate capacity in detail subbands
    capacity = 0
    for detail in coeffs[1:]:
        for band in detail:
            capacity += band.size
    print("Total embedding capacity:", capacity)
    
    # Load secret
    secret_img = Image.open(secret_image_path).convert('L')
    secret_np = np.array(secret_img, dtype=np.uint8)
    secret_pixels = secret_np.size
    
    # If secret > capacity, resize it
    if secret_pixels > capacity:
        side = int(np.sqrt(capacity))
        secret_img_resized = secret_img.resize((side, side), Image.LANCZOS)
        secret_np = np.array(secret_img_resized, dtype=np.uint8)
        print("Secret image resized to", secret_np.shape)
    
    # Quantize secret & flatten
    secret_quant = quantize_to_n_bits(secret_np, bits_used)
    secret_flat = secret_quant.flatten()
    total_secret = secret_flat.size
    print("Total secret elements:", total_secret)
    
    secret_idx = 0
    
    # Embed across all detail subbands in coeffs[1:]
    for d in range(1, len(coeffs)):
        new_detail = []
        for band in coeffs[d]:
            band_flat = band.flatten()
            band_int = np.rint(band_flat).astype(np.int32)
            for i in range(band_int.size):
                if secret_idx < total_secret:
                    secret_val = secret_flat[secret_idx]
                    secret_idx += 1
                    abs_val = abs(band_int[i])
                    # Clear bits and insert
                    cleared = (abs_val >> bits_used) << bits_used
                    new_abs = cleared | secret_val
                    band_int[i] = -new_abs if band_int[i] < 0 else new_abs
                else:
                    break
            band_modified = band_int.reshape(band.shape).astype(np.float32)
            new_detail.append(band_modified)
        coeffs[d] = tuple(new_detail)
        if secret_idx >= total_secret:
            break
    
    print("Embedded", secret_idx, "secret pixels.")
    
    # Reconstruct stego
    stego_recon = pywt.waverec2(coeffs, wavelet=wavelet)
    stego_recon = np.clip(np.rint(stego_recon), 0, 255).astype(np.uint8)
    
    # Save
    stego_img = Image.fromarray(stego_recon, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego saved to {stego_image_path}.")
    
    return cover_np, secret_np, secret_quant, stego_recon, secret_np.shape

def extract_secret_image_dwt_distributed(
    stego_image_path,
    bits_used=1,           # Must match embedding
    wavelet='bior2.2',     # Must match embedding
    level=1,
    secret_shape=None
):
    """
    Extract the secret image from a stego image where data was distributed
    across the detail subbands in a multi-level (or single-level) DWT.
    """
    stego_img = Image.open(stego_image_path).convert('L')
    stego_np = np.array(stego_img, dtype=np.uint8)
    
    coeffs = pywt.wavedec2(stego_np, wavelet=wavelet, level=level)
    
    extracted_vals = []
    for detail in coeffs[1:]:
        for band in detail:
            band_flat = band.flatten()
            band_int = np.rint(band_flat).astype(np.int32)
            for val in band_int:
                secret_val = abs(val) & ((1 << bits_used) - 1)
                extracted_vals.append(secret_val)
    
    extracted_array = np.array(extracted_vals, dtype=np.uint8)
    
    if secret_shape is not None:
        total_pixels = secret_shape[0] * secret_shape[1]
        extracted_array = extracted_array[:total_pixels]
        secret_extracted_quant = extracted_array.reshape(secret_shape)
    else:
        secret_extracted_quant = extracted_array
    
    secret_extracted = dequantize_from_n_bits(secret_extracted_quant, bits_used)
    return secret_extracted


def apply_otsu_threshold(grayscale_img):
    """
    Compute Otsu's threshold for a grayscale image and binarize it.
    Returns a binary (0, 255) image.
    """
    thresh_val = threshold_otsu(grayscale_img)
    bin_img = (grayscale_img >= thresh_val).astype(np.uint8) * 255
    return bin_img

def morphological_cleanup(bin_img, radius_open=1, radius_close=1):
    """
    Perform a morphological 'opening' (erosion followed by dilation)
    then 'closing' (dilation followed by erosion)
    to remove small specks in a binary image.
    """
    # skimage.morphology expects boolean images for morphological ops
    bin_bool = (bin_img > 0)
    opened = opening(bin_bool, disk(radius_open))
    cleaned = closing(opened, disk(radius_close))
    return (cleaned.astype(np.uint8)) * 255

def main(cover_image_path, secret_image_path):
    stego_image_path = "stego_singlelevel_bior2.2.png"
    bits_used = 2
    wavelet = "bior2.2"
    decomposition_level = 3
    cover_np, secret_np, secret_quant, stego_np, secret_shape = embed_secret_image_dwt_distributed(
        cover_image_path,
        secret_image_path,
        stego_image_path,
        bits_used=bits_used,
        wavelet=wavelet,
        level=decomposition_level
    )
    secret_extracted = extract_secret_image_dwt_distributed(
        stego_image_path,
        bits_used=bits_used,
        wavelet=wavelet,
        level=decomposition_level,
        secret_shape=secret_shape
    )
    secret_extracted_otsu = apply_otsu_threshold(secret_extracted)
    secret_extracted_clean = morphological_cleanup(secret_extracted_otsu)
    embedTime = timeit.timeit(
        lambda: embed_secret_image_dwt_distributed(
            cover_image_path, secret_image_path, stego_image_path, bits_used=bits_used, wavelet=wavelet, level=decomposition_level
        ),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_secret_image_dwt_distributed(
            stego_image_path, bits_used=bits_used, wavelet=wavelet, level=decomposition_level, secret_shape=secret_shape
        ),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, stego_np)
    secret_dequant = dequantize_from_n_bits(secret_quant, bits_used)
    mse_secret = compute_mse(secret_dequant, secret_extracted_clean)
    psnr_secret = compute_psnr(mse_secret)
    ssim_secret = compute_ssim(secret_dequant, secret_extracted_clean)
    
    # -------------------------------------------------------------------------
    # 5) DISPLAY RESULTS
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 14))
    
    # Row 1: Cover vs. Stego
    axes[0, 0].imshow(cover_np, cmap='gray')
    axes[0, 0].set_title("Cover Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(stego_np, cmap='gray')
    axes[0, 1].set_title("Stego Image (Single-Level, bior2.2)")
    axes[0, 1].axis("off")
    
    # Row 2: Original Secret vs. Extracted
    axes[0, 2].imshow(secret_np, cmap='gray')
    axes[0, 2].set_title("Original Secret (resized if needed)")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(secret_extracted, cmap='gray')
    axes[1, 0].set_title("Extracted Secret (Grayscale)")
    axes[1, 0].axis("off")
    
    # Row 3: Otsu & Cleaned
    axes[1, 1].imshow(secret_extracted_otsu, cmap='gray')
    axes[1, 1].set_title("Otsu Binarized Secret")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(secret_extracted_clean, cmap='gray')
    axes[1, 2].set_title("Cleaned (Morphological Opening)")
    axes[1, 2].axis("off")
    
    fig.suptitle(
        f"Cover vs. Stego: MSE={mse_cover:.2f}, PSNR={psnr_cover:.2f} dB, SSIM={ssim_cover:.4f}\n"
        f"Secret vs. Extracted: MSE={mse_secret:.2f}, PSNR={psnr_secret:.2f} dB, SSIM={ssim_secret:.4f}\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python DWT_otsu.py original_image secret_image")
        sys.exit(1)

    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]

    print("Ricevuto:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#multilevel + otsu nulla di che :)