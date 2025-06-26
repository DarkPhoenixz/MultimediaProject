#Matteo Steganografia
import timeit
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Uso: python DCT-full.py original_image secret_image")
    sys.exit(1)

cover_image_path = sys.argv[1]
secret_image_path = sys.argv[2]

print("Ricevuto:")
print(" - image:", cover_image_path)
print(" - timageext :", secret_image_path)

def quantize_to_n_bits(image_array, n):
    """
    Quantize an 8-bit grayscale image to n bits.
    For n=2, the image will have 2^2 = 4 gray levels.
    """
    levels = 2 ** n
    quantized = (image_array.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)
    return quantized

def dequantize_from_n_bits(quantized_array, n):
    """
    Scale a quantized image (n bits) back to the 8-bit range.
    For n=2, maps 0 -> 0, 1 -> 85, 2 -> 170, 3 -> 255.
    """
    levels = 2 ** n
    dequantized = (quantized_array.astype(np.uint16) * 255 // (levels - 1)).astype(np.uint8)
    return dequantized

def embed_secret_image_dwt(cover_image_path, secret_image_path, stego_image_path, bits_used=2, wavelet='haar'):
    """
    Embed a secret image into a cover image using DWT-based steganography.
    
    Steps:
    1. Perform one-level DWT on the cover image.
    2. Resize and quantize the secret image to match one of the subbands (here, cH).
    3. Replace the lower 'bits_used' bits of each coefficient in cH with the secret data.
    4. Reconstruct the stego image with inverse DWT.
    
    Both images must be grayscale.
    """
    # Load the cover image in grayscale.
    cover_img = Image.open(cover_image_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.uint8)
    
    # Perform one-level DWT on the cover image.
    coeffs = pywt.dwt2(cover_np, wavelet)
    cA, (cH, cV, cD) = coeffs  # cA, cH, cV, cD are all 2D arrays.
    
    # Load the secret image in grayscale.
    secret_img = Image.open(secret_image_path).convert('L')
    # Resize secret image to match cH's dimensions.
    secret_img_resized = secret_img.resize((cH.shape[1], cH.shape[0]), Image.LANCZOS)
    secret_np = np.array(secret_img_resized, dtype=np.uint8)
    
    # Quantize the secret image to n bits.
    secret_quant = quantize_to_n_bits(secret_np, bits_used)
    
    # Embed secret into cH subband.
    # Convert cH coefficients to integers.
    cH_int = np.rint(cH).astype(np.int32)
    
    # Use vectorized operations:
    abs_cH = np.abs(cH_int)
    # Clear the lower bits.
    cleared = (abs_cH >> bits_used) << bits_used
    # Insert secret bits.
    new_abs = cleared | secret_quant
    # Restore the original sign (for zero, assume non-negative).
    new_cH_int = np.where(cH_int < 0, -new_abs, new_abs)
    
    # Replace the horizontal detail coefficients with the modified values.
    cH_modified = new_cH_int.astype(np.float32)
    
    # Reconstruct the stego image using the modified coefficients.
    coeffs_modified = (cA, (cH_modified, cV, cD))
    stego_recon = pywt.idwt2(coeffs_modified, wavelet)
    stego_recon = np.clip(np.rint(stego_recon), 0, 255).astype(np.uint8)
    
    # Save the stego image.
    stego_img = Image.fromarray(stego_recon, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path} using DWT steganography.")
    
    return cover_np, secret_np, secret_quant, stego_recon

def extract_secret_image_dwt(stego_image_path, bits_used=2, wavelet='haar'):
    """
    Extract the secret image from the stego image (using DWT).
    
    The process:
    1. Perform one-level DWT on the stego image.
    2. Retrieve the modified subband (cH) and extract the lower 'bits_used' bits.
    3. Dequantize the extracted secret to obtain the recovered secret image.
    
    Note: The recovered secret image will have the same dimensions as the chosen subband.
    """
    stego_img = Image.open(stego_image_path).convert('L')
    stego_np = np.array(stego_img, dtype=np.uint8)
    
    # Perform DWT on the stego image.
    coeffs = pywt.dwt2(stego_np, wavelet)
    _, (cH, _, _) = coeffs
    
    # Convert cH coefficients to integers.
    cH_int = np.rint(cH).astype(np.int32)
    # Extract the secret bits from the absolute value.
    secret_extracted_quant = np.abs(cH_int) & ((1 << bits_used) - 1)
    
    # Dequantize back to 8-bit range.
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

# --- Main Execution ---
if __name__ == "__main__":
    stego_image_path = 'stego_dwt.png'   # Output stego image file
    
    bits_used = 2  # Number of bits to embed into the chosen subband
    
    # Embed the secret image using DWT-based steganography.
    cover_np, secret_np, secret_quant, stego_np = embed_secret_image_dwt(
        cover_image_path, secret_image_path, stego_image_path, bits_used=bits_used)
    
    # Extract the secret image.
    secret_extracted = extract_secret_image_dwt(stego_image_path, bits_used=bits_used)
    
    # Calculate embedding computational time
    embedTime = timeit.timeit(
        "embed_secret_image_dwt(cover_image_path, secret_image_path, stego_image_path, bits_used=bits_used)",
        globals=globals(),
        number=10
    )

    # Calculate extraction computational time
    extractTime = timeit.timeit(
        "extract_secret_image_dwt(stego_image_path, bits_used=bits_used)",
        globals=globals(),
        number=10
    )

    # Compute MSE and PSNR for cover vs. stego (to assess imperceptibility).
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    
    # Compute MSE and PSNR for the secret image.
    # Note: secret_np is resized (and quantized) during embedding.
    secret_dequant = dequantize_from_n_bits(secret_quant, bits_used)
    mse_secret = compute_mse(secret_dequant, secret_extracted)
    psnr_secret = compute_psnr(mse_secret)
    
    # Display results.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cover_np, cmap='gray')
    axes[0, 0].set_title("Cover Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(stego_np, cmap='gray')
    axes[0, 1].set_title("Stego Image (DWT Domain)")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(secret_np, cmap='gray')
    axes[1, 0].set_title("Original (Resized) Secret Image")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(secret_extracted, cmap='gray')
    axes[1, 1].set_title("Extracted Secret Image")
    axes[1, 1].axis("off")
    
    fig.suptitle(
        f"Cover vs. Stego MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB\n"
        f"Secret (quantized) vs. Extracted MSE: {mse_secret:.2f}, PSNR: {psnr_secret:.2f} dB\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    
    plt.tight_layout()
    plt.show()

"""
Le modifiche sono più diffuse e meno visibili come pattern regolari (niente aliasing da sottocampionamento progressivo)

Quindi, niente bande regolari → solo rumore distribuito più omogeneamente
"""