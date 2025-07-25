#Grazy  Steganography
import sys
import numpy as np
import timeit
from PIL import Image
import matplotlib.pyplot as plt

def pad_to_shape(img_np, target_shape):
    """
    Pad a numpy array to the target shape with zeros.
    """
    padded = np.zeros(target_shape, dtype=img_np.dtype)
    h, w = img_np.shape
    padded[:h, :w] = img_np
    return padded

def quantize_to_n_bits(image_array, n):
    """
    Quantize an 8-bit grayscale image to n bits.
    For n=2, the image will have 2^2 = 4 gray levels.
    """
    levels = 2 ** n
    # Scale pixels from 0-255 to 0-(levels-1)
    quantized = (image_array.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)
    return quantized

def dequantize_from_n_bits(quantized_array, n):
    """
    Scale a quantized image (n bits) back to 8-bit range.
    For n=2, maps 0->0, 1->85, 2->170, 3->255.
    """
    levels = 2 ** n
    dequantized = (quantized_array.astype(np.uint16) * 255 // (levels - 1)).astype(np.uint8)
    return dequantized

def embed_secret_image(cover_np, secret_np, bits_used):
    """
    Embed a secret image into a cover image using the specified number of LSBs.
    - Both images are assumed to be 512x512 and in grayscale.
    - The secret image is quantized to 'bits_used' bits.
    - The cover image's lower 'bits_used' bits are replaced by the secret image's data.
    """
    # Ensure images have the same dimensions
    if cover_np.shape < secret_np.shape:
        padded = np.zeros(secret_np.shape, dtype=cover_np.dtype)
        h, w = cover_np.shape
        padded[:h, :w] = cover_np
    
    # Quantize the secret image to the desired bit-depth.
    secret_quant = quantize_to_n_bits(secret_np, bits_used)
    
    # Create a mask to clear the lower 'bits_used' bits in the cover image.
    # For bits_used=2, this mask should be 0xFC (252 in decimal).
    mask = (~((1 << bits_used) - 1)) & 0xFF
    cover_cleared = cover_np & mask
    
    # Embed the secret by OR-ing the cleared cover with the secret quantized data.
    stego_np = cover_cleared | secret_quant
    
    return cover_np, secret_np, secret_quant, stego_np

def extract_secret_image(stego_np, bits_used):
    """
    Extract the secret image from the stego image by retrieving the LSBs.
    The extracted secret is still in its quantized form (n bits); we then scale it back to 8-bit.
    """
    
    # Extract the secret data: get the 'bits_used' LSBs.
    secret_extracted_quant = stego_np & ((1 << bits_used) - 1)
    
    # Scale back to 8-bit for display.
    secret_extracted = dequantize_from_n_bits(secret_extracted_quant, bits_used)
    
    return secret_extracted

def compute_mse(imageA, imageB):
    """Compute the Mean Squared Error between two images."""
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def compute_psnr(mse, max_pixel=255.0):
    """Compute the PSNR (Peak Signal-to-Noise Ratio) given the MSE."""
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Compute SSIM (Structural Similarity Index) between two images.
    """
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
    print("Received:")
    print(" - image:", cover_image_path)
    print(" - logo :", secret_image_path)
    if not cover_image_path:
        print("No image selected. Exiting program.")
        exit()
    cover_image = Image.open(cover_image_path).convert("L")
    cover_np = np.array(cover_image)
    h_cover, w_cover = cover_np.shape
    if not secret_image_path:
        print("No secret image selected. Exiting program.")
        exit()
    secret_image = Image.open(secret_image_path).convert("L")
    secret_np = np.array(secret_image)
    h_secret, w_secret = secret_np.shape
    if h_secret > h_cover or w_secret > w_cover:
        print(f"Resizing secret image from {(w_secret, h_secret)} to {(w_cover, h_cover)}")
        secret_image = secret_image.resize((w_cover, h_cover))
        secret_np = np.array(secret_image)
    elif h_secret < h_cover or w_secret < w_cover:
        print(f"Padding secret image from {(w_secret, h_secret)} to {(w_cover, h_cover)}")
        secret_np = pad_to_shape(secret_np, (h_cover, w_cover))
    stego_image_path = 'stego.png'
    bits_used = 2
    cover_np, secret_np, secret_quant, stego_np = embed_secret_image(
        cover_np, secret_np, bits_used=bits_used)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path}")
    secret_extracted = extract_secret_image(stego_np, bits_used=bits_used)
    embedTime = timeit.timeit(
        lambda: embed_secret_image(cover_np, secret_np, bits_used=bits_used),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_secret_image(stego_np, bits_used=bits_used),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, stego_np)
    secret_dequant = dequantize_from_n_bits(secret_quant, bits_used)
    mse_secret = compute_mse(secret_dequant, secret_extracted)
    psnr_secret = compute_psnr(mse_secret)
    ssim_secret = compute_ssim(secret_dequant, secret_extracted)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cover_np, cmap='gray')
    axes[0, 0].set_title("Cover Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(stego_np, cmap='gray')
    axes[0, 1].set_title("Stego Image")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(secret_dequant, cmap='gray')
    axes[1, 0].set_title("Original Secret Image")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(secret_extracted, cmap='gray')
    axes[1, 1].set_title("Extracted Secret Image")
    axes[1, 1].axis("off")
    fig.suptitle(
        f"Original vs. Stego Image MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB, SSIM: {ssim_cover:.4f}\n"
        f"Original vs. Extracted Secret MSE: {mse_secret:.2f}, PSNR: {psnr_secret:.2f} dB, SSIM: {ssim_secret:.4f}\n"
        f"Embedding Time: {embedTime:.5f}s, Extraction Time: {extractTime:.5f}s",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python LSB_spatial.py image_path logo_path")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]
    main(cover_image_path, secret_image_path)

"""
Automatic resizing to image size
No limitation on image size
"""