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
    """Calcola la DCT 2D (tipo II) con normalizzazione ortogonale."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Calcola l'IDCT 2D (tipo III) con normalizzazione ortogonale."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_bits_in_quant_coeff(coef, bits, num_bits):
    """
    Inserisce 'num_bits' (fornite come stringa, ad es. "10") nei LSB di un coefficiente quantizzato.
    Il coefficiente è un intero (dopo il rounding).
    """
    coef_int = int(coef)
    mask = ~((1 << num_bits) - 1)  # per num_bits=2 => mask = ~0b11 = 0xFC
    coef_cleared = coef_int & mask
    new_coef = coef_cleared | int(bits, 2)
    return new_coef

def extract_bits_from_quant_coeff(coef, num_bits):
    """
    Estrae i 'num_bits' LSB da un coefficiente quantizzato (intero) e restituisce la stringa corrispondente.
    """
    coef_int = int(coef)
    mask = (1 << num_bits) - 1
    bits_val = coef_int & mask
    return format(bits_val, f'0{num_bits}b')

def embed_secret_dct(cover_path, secret_path, stego_path, block_size, 
                     quant_matrix, target_coeff, bits_per_block):
    """
    - Immagine cover: 512x512 in scala di grigi.
    - Immagine segreta: viene ridimensionata a 32x32 (1024 pixel → 8192 bit).
    - Dividiamo il cover in blocchi 8x8 (4096 blocchi) e embeddiamo in ogni blocco 'bits_per_block' bit,
      inserendoli nel coefficiente quantizzato in posizione target (ad es. (3,3)).
    - Al contrario dell'approccio JPEG-like, qui quantizziamo solo il coefficiente target,
      preservando il resto del blocco e riducendo distorsioni.
    """
    # Carica cover in scala di grigi
    cover_img = Image.open(cover_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.float32)
    H, W = cover_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz  # 4096 blocchi (512x512 / 8x8)

    # Prepara l'immagine segreta: ridimensiona a 32x32
    secret_img = Image.open(secret_path).convert('L')
    secret_resized = secret_img.resize((16,16))
    secret_np = np.array(secret_resized, dtype=np.uint8)
    # Converti l'immagine segreta in una stringa di bit (8 bit per pixel → 1024*8 = 8192 bit)
    secret_bits = ''.join(format(p, '08b') for p in secret_np.flatten())
    print(bits_per_block)
    if len(secret_bits) != total_blocks * bits_per_block:
        secret_bits = secret_bits.ljust(total_blocks * bits_per_block, '0')
        # raise ValueError(f"La lunghezza della bitstream segreta ({len(secret_bits)} bit) "
        #                  f"non corrisponde alla capacità ({total_blocks * bits_per_block} bit).")

    bit_index = 0
    stego_np = np.zeros_like(cover_np)

    # Per ogni blocco 8x8
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = cover_np[r:r+block_size, c:c+block_size]

            # 1) DCT del blocco
            dct_block = dct2(block)

            # 2) SOLO quantizzazione del coefficiente target
            rr, cc = target_coeff
            orig_coef_val = dct_block[rr, cc]

            # Dividiamo per la matrice e arrotondiamo, ma SOLO per il coeff. target
            q_val = orig_coef_val / quant_matrix[rr, cc]
            q_val_rounded = int(round(q_val))

            # 3) Embedding: prendi i 'bits_per_block' bit dalla secret_bits
            bits_to_embed = secret_bits[bit_index:bit_index + bits_per_block]
            bit_index += bits_per_block

            # Embed dei bit nei LSB del coeff. quantizzato
            new_q_val = embed_bits_in_quant_coeff(q_val_rounded, bits_to_embed, num_bits=bits_per_block)

            # 4) Dequantizzazione SOLO del coeff. target
            dct_block[rr, cc] = new_q_val * quant_matrix[rr, cc]

            # 5) IDCT per ricostruire il blocco
            stego_block = idct2(dct_block)
            stego_np[r:r+block_size, c:c+block_size] = stego_block

    # Clip e conversione in 8-bit
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_path)
    print(f"Immagine stego salvata in: {stego_path}")
    return cover_np.astype(np.uint8), secret_np, stego_np

def extract_secret_dct(stego_path, block_size, quant_matrix, 
                       target_coeff, bits_per_block, secret_size):
    """
    Estrae i bit da ogni blocco (dalla stessa posizione target) e ricostruisce l'immagine segreta.
    secret_size è (altezza, larghezza) in pixel; in questo esempio 32x32 (1024 pixel → 8192 bit).
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

            # Quantizziamo SOLO il coeff. target
            rr, cc = target_coeff
            coef_val = dct_block[rr, cc]
            q_val = int(round(coef_val / quant_matrix[rr, cc]))

            # Estrazione dei bit
            bits = extract_bits_from_quant_coeff(q_val, num_bits=bits_per_block)
            extracted_bits += bits

    # Dovrebbero essere esattamente secret_size[0]*secret_size[1]*8 bit
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
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=window_size)

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
        print("Uso: python DCT_singleQuant.py original_image secret_image")
        sys.exit(1)

    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]

    print("Ricevuto:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#spiegare perchè limmagine è grande 32x32