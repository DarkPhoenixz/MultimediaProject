#Emy Steganografia 
import timeit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import sys


###############################################################################
# Matrice di quantizzazione JPEG 
###############################################################################
JPEG_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
], dtype=np.float32)

def dct2(block):
    """Calculates the 2D DCT (type II) with orthogonal normalization."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Calculates the 2D IDCT (type III) with orthogonal normalization."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_bits_in_quant_coeff(coef, bits, num_bits):
    """
    Inserts 'num_bits' (provided as a string, e.g., "10") into the LSBs of a quantized coefficient.
    The coefficient is an integer (after rounding).
    """
    coef_int = int(coef)
    mask = ~((1 << num_bits) - 1)  # for num_bits=2 => mask = ~0b11 = 0xFC
    coef_cleared = coef_int & mask
    new_coef = coef_cleared | int(bits, 2)
    return new_coef

def extract_bits_from_quant_coeff(coef, num_bits):
    """
    Extracts the 'num_bits' LSBs from a quantized coefficient (integer) and returns the corresponding string.
    """
    coef_int = int(coef)
    mask = (1 << num_bits) - 1
    bits_val = coef_int & mask
    return format(bits_val, f'0{num_bits}b')

def embed_secret_dct(cover_path, secret_path, stego_path, block_size, 
                     quant_matrix, target_coeff, bits_per_block):
    """
    - Cover image: 512x512 in grayscale.
    - Secret image: is resized to 32x32 (1024 pixels → 8192 bits).
    - We divide the cover into 8x8 blocks (4096 blocks) and embed 'bits_per_block' bits in each block,
      embedding them in the quantized coefficient at position target (e.g., (3,3)).
    - Unlike the JPEG-like approach, here we only quantize the target coefficient,
      preserving the rest of the block and reducing distortion.
    """
    # Load cover in grayscale
    cover_img = Image.open(cover_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.float32)
    H, W = cover_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz  # 4096 blocks (512x512 / 8x8)

    # Prepare secret image: resize to 32x32
    secret_img = Image.open(secret_path).convert('L')
    secret_resized = secret_img.resize((16,16))
    secret_np = np.array(secret_resized, dtype=np.uint8)
    # Convert secret image to a bit string (8 bits per pixel → 1024*8 = 8192 bits)
    secret_bits = ''.join(format(p, '08b') for p in secret_np.flatten())
    print(bits_per_block)
    if len(secret_bits) != total_blocks * bits_per_block:
        secret_bits = secret_bits.ljust(total_blocks * bits_per_block, '0')
        # raise ValueError(f"The length of the secret bitstream ({len(secret_bits)} bits) "
        #                  f"does not match the capacity ({total_blocks * bits_per_block} bits).")

    bit_index = 0
    stego_np = np.zeros_like(cover_np)

    # For each 8x8 block
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = cover_np[r:r+block_size, c:c+block_size]

            # 1) DCT of the block
            dct_block = dct2(block)

            # 2) ONLY quantization of the target coefficient
            rr, cc = target_coeff
            orig_coef_val = dct_block[rr, cc]

            # Divide by the matrix and round, but ONLY for the target coefficient
            q_val = orig_coef_val / quant_matrix[rr, cc]
            q_val_rounded = int(round(q_val))

            # 3) Embedding: take the 'bits_per_block' bits from secret_bits
            bits_to_embed = secret_bits[bit_index:bit_index + bits_per_block]
            bit_index += bits_per_block

            # Embed the bits into the LSBs of the quantized coefficient
            new_q_val = embed_bits_in_quant_coeff(q_val_rounded, bits_to_embed, num_bits=bits_per_block)

            # 4) Dequantization ONLY of the target coefficient
            dct_block[rr, cc] = new_q_val * quant_matrix[rr, cc]

            # 5) IDCT to reconstruct the block
            stego_block = idct2(dct_block)
            stego_np[r:r+block_size, c:c+block_size] = stego_block

    # Clip and convert to 8-bit
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_path)
    print(f"Stego image saved in: {stego_path}")
    return cover_np.astype(np.uint8), secret_np, stego_np

def extract_secret_dct(stego_path, block_size, quant_matrix, 
                       target_coeff, bits_per_block, secret_size):
    """
    Extracts bits from each block (at the same target position) and reconstructs the secret image.
    secret_size is (height, width) in pixels; in this example 32x32 (1024 pixels → 8192 bits).
    """
    stego_img = Image.open(stego_path).convert('L')
    stego_np = np.array(stego_img, dtype=np.float32)
    H, W = stego_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size

    extracted_bits = ""
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = stego_np[r:r+block_size, c:c+block_size]

            # DCT
            dct_block = dct2(block)

            # Quantization ONLY of the coefficient
            rr, cc = target_coeff
            coef_val = dct_block[rr, cc]
            q_val = int(round(coef_val / quant_matrix[rr, cc]))

            # Extraction of bits
            bits = extract_bits_from_quant_coeff(q_val, num_bits=bits_per_block)
            extracted_bits += bits

    # Should be exactly secret_size[0]*secret_size[1]*8 bits
    needed_bits = secret_size[0] * secret_size[1] * 8
    extracted_bits = extracted_bits[:needed_bits]

    secret_bytes = []
    for i in range(0, len(extracted_bits), 8):
        byte_str = extracted_bits[i:i+8]
        secret_bytes.append(int(byte_str, 2))
    secret_array = np.array(secret_bytes, dtype=np.uint8).reshape(secret_size)
    secret_img = Image.fromarray(secret_array, mode='L')
    return secret_array, secret_img

def compute_mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def compute_psnr(mse, max_pixel):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel**2) / mse)

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
    cover_np, secret_32x32_np, stego_np = embed_secret_dct(
        cover_image_path, secret_image_path, stego_image_path,
        block_size=8, quant_matrix=JPEG_QUANT_MATRIX, target_coeff=(7,7), bits_per_block=1
    )
    extracted_secret_np, extracted_secret_img = extract_secret_dct(
        stego_image_path, block_size=8, quant_matrix=JPEG_QUANT_MATRIX,
        target_coeff=(7,7), bits_per_block=1, secret_size=(16,16)
    )
    embedTime = timeit.timeit(
        lambda: embed_secret_dct(cover_image_path, secret_image_path, stego_image_path, block_size=8, quant_matrix=JPEG_QUANT_MATRIX, target_coeff=(7,7), bits_per_block=1),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_secret_dct(stego_image_path, block_size=8, quant_matrix=JPEG_QUANT_MATRIX, target_coeff=(7,7), bits_per_block=1, secret_size=(16,16)),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover, max_pixel=255)
    ssim_cover = compute_ssim(cover_np, stego_np)
    mse_secret = compute_mse(secret_32x32_np, extracted_secret_np)
    psnr_secret = compute_psnr(mse_secret, max_pixel=255)
    ssim_secret = compute_ssim(secret_32x32_np, extracted_secret_np)
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes[0,0].imshow(cover_np, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title("Cover Image (512x512)")
    axes[0,0].axis('off')
    axes[0,1].imshow(stego_np, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title("Stego Image (DCT embedding)")
    axes[0,1].axis('off')
    axes[1,0].imshow(secret_32x32_np, cmap='gray')
    axes[1,0].set_title("Original Secret Image (32x32)")
    axes[1,0].axis('off')
    axes[1,1].imshow(extracted_secret_np, cmap='gray')
    axes[1,1].set_title("Extracted Secret Image")
    axes[1,1].axis('off')
    fig.suptitle(
        f"Cover vs Stego: MSE = {mse_cover:.2f}, PSNR = {psnr_cover:.2f} dB, SSIM = {ssim_cover:.4f}\n" +
        f"Secret vs Extracted: MSE = {mse_secret:.2f}, PSNR = {psnr_secret:.2f} dB, SSIM = {ssim_secret:.4f}\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python DCT_singleQuant.py original_image secret_image")
        sys.exit(1)

    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]

    print("Received:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#explain why the image is large 32x32