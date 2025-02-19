import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

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
    For n=2, maps 0->0, 1->85, 2->170, 3->255.
    """
    levels = 2 ** n
    dequantized = (quantized_array.astype(np.uint16) * 255 // (levels - 1)).astype(np.uint8)
    return dequantized

def dct2(block):
    """Compute the 2D Discrete Cosine Transform."""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(block):
    """Compute the 2D Inverse Discrete Cosine Transform."""
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def embed_secret_image_dct(cover_image_path, secret_image_path, stego_image_path, bits_used=2, block_size=8):
    """
    Embed a secret image into a cover image in the DCT domain.
    
    The cover image is divided into blocks (default 8x8). For each block, its DCT is computed,
    each coefficient (rounded to an integer) is modified by replacing its lower 'bits_used'
    bits with secret data, and then the inverse DCT is applied.
    
    Both images must be the same size. The secret image is quantized to n bits.
    """
    # Load images as grayscale
    cover_img = Image.open(cover_image_path).convert('L')
    secret_img = Image.open(secret_image_path).convert('L')
    
    cover_np = np.array(cover_img, dtype=np.uint8)
    secret_np = np.array(secret_img, dtype=np.uint8)
    
    if cover_np.shape != secret_np.shape:
        raise ValueError("Cover and secret images must have the same dimensions.")
    
    # Quantize the secret image to n bits.
    secret_quant = quantize_to_n_bits(secret_np, bits_used)
    
    # Flatten the secret for sequential embedding.
    secret_flat = secret_quant.flatten()
    
    # Prepare an array for the stego image (we use float32 for DCT processing)
    stego_np = np.zeros_like(cover_np, dtype=np.float32)
    
    idx = 0
    rows, cols = cover_np.shape
    # Process the cover image block by block.
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = cover_np[i:i+block_size, j:j+block_size].astype(np.float32)
            # Compute DCT of the block.
            block_dct = dct2(block)
            # Round the DCT coefficients to integers.
            block_dct_int = np.rint(block_dct).astype(np.int32)
            
            # Embed secret data into every coefficient (row-major order).
            for r in range(block_size):
                for c in range(block_size):
                    coeff = block_dct_int[r, c]
                    secret_val = int(secret_flat[idx])
                    idx += 1
                    
                    # Work with the absolute value for bit-level operations.
                    coeff_abs = abs(coeff)
                    # Clear the lower bits and insert the secret value.
                    coeff_cleared = (coeff_abs >> bits_used) << bits_used
                    new_coeff_abs = coeff_cleared | secret_val
                    # Restore the original sign.
                    new_coeff = -new_coeff_abs if coeff < 0 else new_coeff_abs
                    block_dct_int[r, c] = new_coeff
            
            # Convert back to float and perform inverse DCT.
            block_dct_modified = block_dct_int.astype(np.float32)
            block_stego = idct2(block_dct_modified)
            # Clip to valid 8-bit range and assign the block.
            stego_np[i:i+block_size, j:j+block_size] = np.clip(np.rint(block_stego), 0, 255)
    
    stego_np_uint8 = stego_np.astype(np.uint8)
    stego_img = Image.fromarray(stego_np_uint8, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path} using DCT steganography.")
    
    return cover_np, secret_np, secret_quant, stego_np_uint8

def extract_secret_image_dct(stego_image_path, bits_used=2, block_size=8):
    """
    Extract the secret image from the stego image (in the DCT domain).
    
    The stego image is processed block by block; for each block the DCT is computed and the
    lower 'bits_used' bits (from the absolute value of each coefficient) are retrieved.
    The result is then dequantized to form the final secret image.
    """
    stego_img = Image.open(stego_image_path).convert('L')
    stego_np = np.array(stego_img, dtype=np.uint8)
    
    rows, cols = stego_np.shape
    secret_flat = np.zeros(rows * cols, dtype=np.uint8)
    
    idx = 0
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = stego_np[i:i+block_size, j:j+block_size].astype(np.float32)
            block_dct = dct2(block)
            block_dct_int = np.rint(block_dct).astype(np.int32)
            for r in range(block_size):
                for c in range(block_size):
                    coeff = block_dct_int[r, c]
                    # Extract secret bits from the absolute value.
                    secret_val = abs(coeff) & ((1 << bits_used) - 1)
                    secret_flat[idx] = secret_val
                    idx += 1
    
    secret_quant_extracted = secret_flat.reshape(stego_np.shape)
    secret_extracted = dequantize_from_n_bits(secret_quant_extracted, bits_used)
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
    # Update the file paths as needed.
    cover_image_path = 'cover.png'      # 512x512 cover image
    secret_image_path = 'secret.png'    # 512x512 secret image
    stego_image_path = 'stego_dct.png'    # Output stego image file
    
    bits_used = 2  # Number of bits used for embedding per coefficient
    
    # Embed secret image using DCT-based steganography.
    cover_np, secret_np, secret_quant, stego_np = embed_secret_image_dct(
        cover_image_path, secret_image_path, stego_image_path, bits_used=bits_used)
    
    # Extract the secret image.
    secret_extracted = extract_secret_image_dct(stego_image_path, bits_used=bits_used)
    
    # Compute MSE and PSNR between the cover and stego images.
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    
    # Compute MSE and PSNR between the quantized secret and the extracted secret.
    secret_dequant = dequantize_from_n_bits(secret_quant, bits_used)
    mse_secret = compute_mse(secret_dequant, secret_extracted)
    psnr_secret = compute_psnr(mse_secret)
    
    # Display the results.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(cover_np, cmap='gray')
    axes[0,0].set_title("Cover Image")
    axes[0,0].axis("off")
    
    axes[0,1].imshow(stego_np, cmap='gray')
    axes[0,1].set_title("Stego Image (DCT Domain)")
    axes[0,1].axis("off")
    
    axes[1,0].imshow(secret_np, cmap='gray')
    axes[1,0].set_title("Original Secret Image")
    axes[1,0].axis("off")
    
    axes[1,1].imshow(secret_extracted, cmap='gray')
    axes[1,1].set_title("Extracted Secret Image")
    axes[1,1].axis("off")
    
    fig.suptitle(
        f"Cover vs. Stego MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB\n"
        f"Secret (quantized) vs. Extracted MSE: {mse_secret:.2f}, PSNR: {psnr_secret:.2f} dB",
        fontsize=14
    )
    
    plt.tight_layout()
    plt.show()
