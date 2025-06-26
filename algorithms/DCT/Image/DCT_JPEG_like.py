#Emy Steganografia
import timeit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import sys
import math

###############################################################################
# 1) Matrice di quantizzazione JPEG (per la luminanza, qualità ~50)
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
# 2) Funzioni DCT/IDCT 2D (con normalizzazione 'ortho')
###############################################################################
def dct2(block):
    """Calcola la DCT 2D (tipo II) con normalizzazione ortogonale."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Calcola l'IDCT 2D (tipo III) con normalizzazione ortogonale."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

###############################################################################
# 3) Funzioni per l'embedding e l'estrazione di bit in un coefficiente quantizzato
###############################################################################
def embed_bits_in_quant_coeff(coef, bits, num_bits=2):
    """
    Inserisce 'num_bits' (fornite come stringa, ad es. "10") nei LSB di un coefficiente quantizzato.
    Il coefficiente è un intero (dopo il rounding).
    """
    coef_int = int(coef)
    mask = ~((1 << num_bits) - 1)  # per num_bits=2 => mask = ~0b11 = 0xFC
    coef_cleared = coef_int & mask
    new_coef = coef_cleared | int(bits, 2)
    return new_coef

def extract_bits_from_quant_coeff(coef, num_bits=2):
    """
    Estrae i 'num_bits' LSB da un coefficiente quantizzato (intero) e restituisce la stringa corrispondente.
    """
    coef_int = int(coef)
    mask = (1 << num_bits) - 1
    bits_val = coef_int & mask
    return format(bits_val, f'0{num_bits}b')

###############################################################################
# 4) Funzione di embedding in dominio DCT
###############################################################################
def embed_secret_dct(cover_path, secret_path, stego_path, block_size, quant_matrix, target_coeff, bits_per_block):
    """
    - Immagine cover: 512x512 in scala di grigi.
    - Immagine segreta: viene ridimensionata a 32x32 (1024 pixel → 8192 bit).
    - Dividiamo il cover in blocchi 8x8 (4096 blocchi) e embeddiamo in ogni blocco 'bits_per_block' bit,
      inserendoli nel coefficiente quantizzato in posizione target (ad es. (3,3)).
    """
    # Carica cover in scala di grigi
    cover_img = Image.open(cover_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.float32)
    H, W = cover_np.shape

    num_blocks_vert = H // block_size
    num_blocks_horiz = W // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz  # 4096 blocchi

    def resizer():
        res = int(math.sqrt((total_blocks * bits_per_block) / block_size))
        print(res)
        secret_img = Image.open(secret_path).convert('L')
        secret_resized = secret_img.resize((res, res))
        secret_np = np.array(secret_resized, dtype=np.uint8)
        return secret_np, res
    
    # # Prepara l'immagine segreta: ridimensiona a 32x32
    # secret_img = Image.open(secret_path).convert('L')
    # secret_resized = secret_img.resize((32,32))
    # secret_np = np.array(secret_resized, dtype=np.uint8)
    # Converti l'immagine segreta in una stringa di bit (8 bit per pixel → 1024*8 = 8192 bit)
    secret_np, secret_size = resizer()
    secret_bits = ''.join(format(p, '08b') for p in secret_np.flatten())
    if len(secret_bits) != total_blocks * bits_per_block:
        raise ValueError(f"La lunghezza della bitstream segreta ({len(secret_bits)} bit) non corrisponde alla capacità ({total_blocks * bits_per_block} bit).")

    bit_index = 0
    stego_np = np.zeros_like(cover_np)

    # Per ogni blocco
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            r = i * block_size
            c = j * block_size
            block = cover_np[r:r+block_size, c:c+block_size]

            # 1) DCT del blocco
            dct_block = dct2(block)
            # 2) Quantizzazione: dividi per la matrice e arrotonda
            quant_block = np.round(dct_block / quant_matrix).astype(np.int32)
            # 3) Embedding: prendi i 'bits_per_block' bit dalla secret_bits
            bits_to_embed = secret_bits[bit_index:bit_index + bits_per_block]
            bit_index += bits_per_block
            rr, cc = target_coeff
            coef = quant_block[rr, cc]
            new_coef = embed_bits_in_quant_coeff(coef, bits_to_embed, num_bits=bits_per_block)
            quant_block[rr, cc] = new_coef
            # 4) Dequantizzazione
            new_dct_block = quant_block * quant_matrix
            # 5) IDCT per ricostruire il blocco
            stego_block = idct2(new_dct_block)
            stego_np[r:r+block_size, c:c+block_size] = stego_block

    # Clip e conversione in 8-bit
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_path)
    print(f"Immagine stego salvata in: {stego_path}")
    return cover_np.astype(np.uint8), secret_np, stego_np, [secret_size, secret_size]

###############################################################################
# 5) Funzione di estrazione in dominio DCT
###############################################################################
def extract_secret_dct(stego_path, block_size, quant_matrix, target_coeff, bits_per_block, secret_size):
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
            quant_block = np.round(dct_block / quant_matrix).astype(np.int32)
            rr, cc = target_coeff
            coef = quant_block[rr, cc]
            bits = extract_bits_from_quant_coeff(coef, num_bits=bits_per_block)
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

###############################################################################
# 6) Funzioni per il calcolo di MSE e PSNR
###############################################################################
def compute_mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def compute_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel**2) / mse)

###############################################################################
# 7) Esempio di utilizzo
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
        "embed_secret_dct(cover_image_path, secret_image_path, stego_image_path,block_size=8, quant_matrix=Q90, target_coeff=(4,4), bits_per_block=2)",
        globals=globals(),
        number=10
    )
    extractTime = timeit.timeit(
        "extract_secret_dct(stego_image_path, block_size=8, quant_matrix=Q90,target_coeff=(4,4), bits_per_block=2, secret_size=(32,32))",
        globals=globals(),
        number=10
    )
    mse_cover = compute_mse(cover_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    mse_secret = compute_mse(secret_np, extracted_secret_np)
    psnr_secret = compute_psnr(mse_secret)
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
        f"Cover vs Stego: MSE = {mse_cover:.2f}, PSNR = {psnr_cover:.2f} dB\n" +
        f"Secret vs Extracted: MSE = {mse_secret:.2f}, PSNR = {psnr_secret:.2f} dB\n"
        f"Embedding Time: {embedTime:.2f} s, Extraction Time: {extractTime:.2f} s",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python DCT_JPEG_like.py original_image secret_image")
        sys.exit(1)
    cover_image_path = sys.argv[1]
    secret_image_path = sys.argv[2]
    print("Ricevuto:")
    print(" - image:", cover_image_path)
    print(" - timageext :", secret_image_path)
    main(cover_image_path, secret_image_path)

#funziona bene con le potenze del 2