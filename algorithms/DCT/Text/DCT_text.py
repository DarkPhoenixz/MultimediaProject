#Emy Steganografia
import timeit
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct



########################################################
# JPEG Quantization Matrix (for luminance, quality ~50)
########################################################
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

def dct2(block):
    """Compute 2D DCT (type-II) with orthogonal normalization."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Compute 2D IDCT (type-III) with orthogonal normalization."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_bits_in_quant_coeff(coef, bits, num_bits=1):
    """
    Embed 'num_bits' into the LSBs of a quantized coefficient.
    The coefficient is an integer.
    """
    coef_int = int(coef)
    mask = ~((1 << num_bits) - 1)  
    coef_cleared = coef_int & mask
    new_coef = coef_cleared | int(bits, 2)
    return new_coef

def extract_bits_from_quant_coeff(coef, num_bits=1):
    """
    Extract the 'num_bits' LSBs from a quantized coefficient and return them as a string.
    """
    coef_int = int(coef)
    mask = (1 << num_bits) - 1
    bits_val = coef_int & mask
    return format(bits_val, f'0{num_bits}b')

def embed_secret_dct_string(cover_np, secret_message, block_size, quant_matrix, target_coeff, bits_per_block):
    """
    - Cover image: expected to be a 512x512 PNG in grayscale.
    - Secret message: an arbitrary text string.
    - A 32-bit header is embedded first to indicate the length of the secret message.
    - The message (header + content) is converted to a bitstream.
    - The cover image is divided into 8x8 blocks and bits_per_block bits are embedded in 
      the target coefficient of each block.
    - If the message bitstream is shorter than the available capacity, it is padded with zeros.
    - By embedding only 1 bit per block and using a higher-frequency coefficient, distortion is reduced.
    """
    # Load cover image in grayscale
    
    H, W = cover_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz  

    # Convert secret message to bytes then to bit string.
    # Use a 32-bit header to store the length (in bytes) of the secret message.
    secret_bytes = secret_message.encode('utf-8')
    message_length = len(secret_bytes)
    header_bits = format(message_length, '032b')
    message_bits = ''.join(format(b, '08b') for b in secret_bytes)
    final_bits = header_bits + message_bits

    capacity = total_blocks * bits_per_block
    if len(final_bits) > capacity:
        raise ValueError(f"Secret message too long: requires {len(final_bits)} bits, but capacity is {capacity} bits.")
    
    # Pad the bitstream to fill the capacity if needed.
    final_bits = final_bits.ljust(capacity, '0')

    bit_index = 0
    stego_np = np.zeros_like(cover_np)

    # Process each block.
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = cover_np[r:r+block_size, c:c+block_size]

            # DCT of the block
            dct_block = dct2(block)
            # Quantization: divide by the quantization matrix and round
            quant_block = np.round(dct_block / quant_matrix).astype(np.int32)
            # Get the bits to embed for this block
            bits_to_embed = final_bits[bit_index:bit_index + bits_per_block]
            bit_index += bits_per_block
            rr, cc = target_coeff
            coef = quant_block[rr, cc]
            new_coef = embed_bits_in_quant_coeff(coef, bits_to_embed, num_bits=bits_per_block)
            quant_block[rr, cc] = new_coef
            # Dequantization
            new_dct_block = quant_block * quant_matrix
            # IDCT to reconstruct the block
            stego_block = idct2(new_dct_block)
            stego_np[r:r+block_size, c:c+block_size] = stego_block

    # Clip values and convert to 8-bit
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    
    return cover_np.astype(np.uint8), stego_np

def extract_secret_dct_string(stego_path, block_size, quant_matrix, target_coeff, bits_per_block):
    """
    Extracts the bits from each block (from the same target coefficient position),
    reads the 32-bit header to know the length (in bytes) of the secret message,
    and reconstructs the secret text message.
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

    # First 32 bits are the header: message length in bytes.
    header_bits = extracted_bits[:32]
    message_length = int(header_bits, 2)
    message_bits_needed = message_length * 8
    message_bits = extracted_bits[32:32+message_bits_needed]

    # Convert bitstream to bytes.
    secret_bytes = []
    for i in range(0, len(message_bits), 8):
        byte_str = message_bits[i:i+8]
        secret_bytes.append(int(byte_str, 2))
    secret_bytes = bytes(secret_bytes)
    secret_message = secret_bytes.decode('utf-8', errors='replace')
    return secret_message

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
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=window_size)

def main(cover_image_path, secret_text):
    stego_image_path = 'stego_dct_string.png'
    if not cover_image_path:
        print("Nessuna immagine selezionata. Uscita dal programma.")
        exit()
    cover_image = Image.open(cover_image_path).convert("L")
    if cover_image.size != (512, 512):
        print(f"Ridimensionamento immagine cover da {cover_image.size} a 512x512")
        cover_image = cover_image.resize((512, 512))
    cover_np = np.array(cover_image)
    if not secret_text:
        print("Nessun testo selezionato. Uscita dal programma.")
        exit()
    secret_message = secret_text
    cover_np, stego_np = embed_secret_dct_string(
        cover_np, 
        secret_message,
        block_size=8, 
        quant_matrix=Q90, 
        target_coeff=(7,7),
        bits_per_block=1
    )
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved at: {stego_image_path}")
    extracted_message = extract_secret_dct_string(
        stego_image_path, 
        block_size=8, 
        quant_matrix=Q90,
        target_coeff=(7,7),
        bits_per_block=1
    )
    embedTime = timeit.timeit(
        lambda: embed_secret_dct_string(cover_np, secret_message, block_size=8, quant_matrix=Q90, target_coeff=(7,7), bits_per_block=1),
        number=10
    )
    extractTime = timeit.timeit(
        lambda: extract_secret_dct_string(stego_image_path, block_size=8, quant_matrix=Q90, target_coeff=(7,7), bits_per_block=1),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    ssim_cover = compute_ssim(cover_np, stego_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cover_np, cmap='gray')
    ax1.set_title('Cover Image')
    ax1.axis('off')
    ax2.imshow(stego_np, cmap='gray')
    ax2.set_title('Stego Image')
    ax2.axis('off')
    fig.suptitle(
        f"Cover vs. Stego MSE: {mse_cover:.2f}, PSNR: {psnr_cover:.2f} dB, SSIM: {ssim_cover:.4f}\n"
        f"Embedding Time: {embedTime:.5f}s, Extraction Time: {extractTime:.5f}s",
        fontsize=14
    )
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python DCT_text.py image_path text_to_hide")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_text = sys.argv[2]
    print("Ricevuto:")
    print(" - image:", cover_image_path)
    print(" - text :", secret_text)
    main(cover_image_path, secret_text)
    
#non va bene con immagini conpattern o aree molto piatte poiche dct modifica inmaniera aggressiva
#possibili cose da fare dithering
#controllare frequenza di embed
