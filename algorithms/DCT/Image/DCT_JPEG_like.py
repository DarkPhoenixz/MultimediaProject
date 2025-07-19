#Emy Steganografia
import timeit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import sys
import math

###############################################################################
# 1) Quantization matrix JPEG (for luminance, quality ~50)
###############################################################################
Q50 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
], dtype=np.float32)

Q90 = np.array([
    [ 4,  3,  3,  4,  5,  9, 11, 13],
    [ 3,  3,  3,  4,  6, 12, 12, 12],
    [ 3,  3,  4,  5,  9, 12, 14, 12],
    [ 3,  4,  5,  6, 11, 18, 17, 13],
    [ 4,  5,  8, 12, 14, 22, 21, 16],
    [ 5,  8, 12, 13, 17, 21, 23, 19],
    [10, 13, 16, 18, 21, 25, 24, 21],
    [15, 19, 20, 20, 23, 20, 21, 20]
], dtype=np.float32)

Q100 = np.ones((8,8), dtype=np.float32)

###############################################################################
# 2) DCT/IDCT 2D functions (with 'ortho' normalization)
###############################################################################
def dct2(block):
    """Calculates the 2D DCT (type II) with orthogonal normalization."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Calculates the 2D IDCT (type III) with orthogonal normalization."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

###############################################################################
# 3) Functions for embedding and extracting bits from a quantized coefficient
###############################################################################
def embed_bits_in_quant_coeff(coef, bits, num_bits=2):
    """
    Inserts 'num_bits' (provided as a string, e.g., "10") into the LSBs of a quantized coefficient.
    The coefficient is an integer (after rounding).
    """
    coef_int = int(coef)
    mask = ~((1 << num_bits) - 1)  # for num_bits=2 => mask = ~0b11 = 0xFC
    coef_cleared = coef_int & mask
    new_coef = coef_cleared | int(bits, 2)
    return new_coef

def extract_bits_from_quant_coeff(coef, num_bits=2):
    """
    Extracts the 'num_bits' LSBs from a quantized coefficient (integer) and returns the corresponding string.
    """
    coef_int = int(coef)
    mask = (1 << num_bits) - 1
    bits_val = coef_int & mask
    return format(bits_val, f'0{num_bits}b')

###############################################################################
# 4) DCT domain embedding function
###############################################################################
def embed_secret_dct(cover_path, secret_path, stego_path, block_size, quant_matrix, target_coeff, bits_per_block):
    """
    - Cover image: 512x512 in grayscale.
    - Secret image: is resized to 32x32 (1024 pixels → 8192 bits).
    - We divide the cover into 8x8 blocks (4096 blocks) and embed 'bits_per_block' bits in each block,
      inserting them into the quantized coefficient at position target (e.g., (3,3)).
    """
    # Load cover in grayscale
    cover_img = Image.open(cover_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.float32)
    H, W = cover_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz  # 4096 blocks

    def resizer():
        res = int(math.sqrt((total_blocks * bits_per_block) / block_size))
        print(res)
        secret_img = Image.open(secret_path).convert('L')
        secret_resized = secret_img.resize((res, res))
        secret_np = np.array(secret_resized, dtype=np.uint8)
        return secret_np, res
    
    # # Prepare the secret image: resize to 32x32
    # secret_img = Image.open(secret_path).convert('L')
    # secret_resized = secret_img.resize((32,32))
    # secret_np = np.array(secret_resized, dtype=np.uint8)
    # Convert the secret image to a bit string (8 bits per pixel → 1024*8 = 8192 bits)
    secret_np, secret_size = resizer()
    secret_bits = ''.join(format(p, '08b') for p in secret_np.flatten())
    if len(secret_bits) != total_blocks * bits_per_block:
        raise ValueError(f"The length of the secret bitstream ({len(secret_bits)} bits) does not match the capacity ({total_blocks * bits_per_block} bits).")

    bit_index = 0
    stego_np = np.zeros_like(cover_np)

    # For each block
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = cover_np[r:r+block_size, c:c+block_size]

            # 1) DCT of the block
            dct_block = dct2(block)
            # 2) Quantization: divide by the matrix and round
            quant_block = np.round(dct_block / quant_matrix).astype(np.int32)
            # 3) Embedding: take the 'bits_per_block' bits from secret_bits
            bits_to_embed = secret_bits[bit_index:bit_index + bits_per_block]
            bit_index += bits_per_block
            rr, cc = target_coeff
            coef = quant_block[rr, cc]
            new_coef = embed_bits_in_quant_coeff(coef, bits_to_embed, num_bits=bits_per_block)
            quant_block[rr, cc] = new_coef
            # 4) Dequantization
            new_dct_block = quant_block * quant_matrix
            # 5) IDCT to reconstruct the block
            stego_block = idct2(new_dct_block)
            stego_np[r:r+block_size, c:c+block_size] = stego_block

    # Clip and convert to 8-bit
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_path)
    print(f"Stego image saved to: {stego_path}")
    return cover_np.astype(np.uint8), secret_np, stego_np, [secret_size, secret_size]

###############################################################################
# 5) DCT domain extraction function
###############################################################################
def extract_secret_dct(stego_path, block_size, quant_matrix, target_coeff, bits_per_block, secret_size):
    """
    Extracts bits from each block (from the same target position) and reconstructs the secret image.
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
            quant_block = np.round(dct_block / quant_matrix).astype(np.int32)
            rr, cc = target_coeff
            coef = quant_block[rr, cc]
            bits = extract_bits_from_quant_coeff(coef, num_bits=bits_per_block)
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

###############################################################################
# 6) Functions for MSE and PSNR calculation
###############################################################################
def compute_mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def compute_psnr(mse, max_pixel=255.0):
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

###############################################################################
# 7) Example usage
###############################################################################
def main(cover_image_path, secret_image_path):
    stego_image_path = 'stego_dct.png'
    cover_np, secret_np, stego_np, size = embed_secret_dct(
        cover_image_path, secret_image_path, stego_image_path,
        block_size=8, quant_matrix=Q90, target_coeff=(4,4), bits_per_block=2
    )
    extracted_secret_np, extracted_secret_img = extract_secret_dct(
        stego_image_path, block_size=8, quant_matrix=Q90,
        target_coeff=(4,4), bits_per_block=2, secret_size=size
    )
    embedTime = timeit.timeit(
        lambda: embed_secret_dct(cover_image_path, secret_image_path, stego_image_path, block_size=8, quant_matrix=Q90, target_coeff=(4,4), bits_per_block=2),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_secret_dct(stego_image_path, block_size=8, quant_matrix=Q90, target_coeff=(4,4), bits_per_block=2, secret_size=size),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, stego_np)
    mse_secret = compute_mse(secret_np, extracted_secret_np)
    psnr_secret = compute_psnr(mse_secret)
    ssim_secret = compute_ssim(secret_np, extracted_secret_np)
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes[0,0].imshow(cover_np, cmap='gray')
    axes[0,0].set_title("Cover Image (512x512)")
    axes[0,0].axis('off')
    axes[0,1].imshow(stego_np, cmap='gray')
    axes[0,1].set_title("Stego Image (DCT embedding)")
    axes[0,1].axis('off')
    axes[1,0].imshow(secret_np, cmap='gray')
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
        print("Usage: python DCT_JPEG_like.py original_image secret_image")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]
    print("Received:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#works well with powers of 2