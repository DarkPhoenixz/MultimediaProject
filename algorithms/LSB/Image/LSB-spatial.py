#Grazy  Steganografia
import sys
import numpy as np
import timeit
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Uso: python LSB-spatial.py image_path logo_path")
    sys.exit(1)

cover_image_path = sys.argv[1]
secret_image_path = sys.argv[2]

print("Ricevuto:")
print(" - image:", cover_image_path)
print(" - logo :", secret_image_path)

def pad_to_shape(img_np, target_shape):
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

if __name__ == "__main__":

    if not cover_image_path:
        print("Nessuna immagine selezionata. Uscita dal programma.")
        exit()

    cover_image = Image.open(cover_image_path).convert("L")
    cover_np = np.array(cover_image)
    h_cover, w_cover = cover_np.shape

    if not secret_image_path:
        print("Nessuna immagine selezionata. Uscita dal programma.")
        exit()

    secret_image = Image.open(secret_image_path).convert("L")
    secret_np = np.array(secret_image)
    h_secret, w_secret = secret_np.shape

    if h_secret > h_cover or w_secret > w_cover:
        print(f"Ridimensionamento secret image da {(w_secret, h_secret)} a {(w_cover, h_cover)}")
        secret_image = secret_image.resize((w_cover, h_cover))
        secret_np = np.array(secret_image)

    elif h_secret < h_cover or w_secret < w_cover:
        print(f"Padding secret image da {(w_secret, h_secret)} a {(w_cover, h_cover)}")
        secret_np = pad_to_shape(secret_np, (h_cover, w_cover))

    # Percorsi hardcoded
    stego_image_path = 'stego.png'
    
    bits_used = 2
    # Embed
    cover_np, secret_np, secret_quant, stego_np = embed_secret_image(
        cover_np, secret_np, bits_used=bits_used)
    
    # Save stego image.
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path}")

    # Extract
    secret_extracted = extract_secret_image(stego_np, bits_used=bits_used)

    # Tempi
    embedTime = timeit.timeit(
        "embed_secret_image(cover_np, secret_np, bits_used=bits_used)",
        globals=globals(),
        number=10
    )
    extractTime = timeit.timeit(
        "extract_secret_image(stego_np, bits_used=bits_used)",
        globals=globals(),
        number=10
    )

    # MSE e PSNR
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)

    secret_dequant = dequantize_from_n_bits(secret_quant, bits_used)
    mse_secret = compute_mse(secret_dequant, secret_extracted)
    psnr_secret = compute_psnr(mse_secret)

    # Display
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
        f"Original vs. Stego Image MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB\n"
        f"Original vs. Extracted Secret MSE: {mse_secret:.2f}, PSNR: {psnr_secret:.2f} dB\n"
        f"Embedding Time: {embedTime:.5f}s, Extraction Time: {extractTime:.5f}s",
        fontsize=14
    )

    plt.tight_layout()

    plt.show()

"""
ridimensionamento automatico a misura immagine
no limitazioni su dimensione immagine
"""