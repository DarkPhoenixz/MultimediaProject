import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def dct2(block):
    """2D DCT (type II) with orthogonal normalization."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """2D inverse DCT (type III) with orthogonal normalization."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_bits_in_coefficient(coef, bits, num_bits=2):
    """
    Embed 'num_bits' (as a string) into the 2's-complement 16-bit representation
    of a DCT coefficient.
    """
    # Round and convert to int16
    coef_int = np.int16(round(coef))
    # Interpret as unsigned 16-bit to do bitwise operations
    coef_uint = np.uint16(coef_int)
    # Create mask to clear lower num_bits bits
    mask = (~((1 << num_bits) - 1)) & 0xFFFF
    coef_cleared = coef_uint & mask
    # Set lower bits to our value
    bit_val = int(bits, 2)
    new_coef_uint = coef_cleared | bit_val
    # Convert back to signed 16-bit and then to float for inverse DCT
    new_coef = np.int16(new_coef_uint)
    return float(new_coef)

def embed_secret_image_dct(cover_image_path, secret_image_path, stego_image_path, bits_per_block=2, target_coeff=(3,3)):
    """
    Embed a secret image into the cover image in the DCT domain.
    
    - The cover image is divided into 8×8 blocks.
    - In each block, 2 bits (default) are embedded into the chosen DCT coefficient (default position (3,3)).
    - Total capacity = (# of blocks) * (bits_per_block). For a 512×512 cover image (4096 blocks) with 2 bits per block,
      the capacity is 8192 bits, i.e. 1024 secret pixels (e.g. a 32×32 secret image).
      
    The secret image is resized to 32×32 before embedding.
    """
    # Load cover image (512x512, grayscale)
    cover_img = Image.open(cover_image_path).convert('L')
    cover_np = np.array(cover_img, dtype=np.float32)
    height, width = cover_np.shape

    # Determine block size and number of blocks
    block_size = 8
    num_blocks_vert = height // block_size
    num_blocks_horiz = width // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz

    # Capacity in bits and required secret image size (in pixels)
    capacity_bits = total_blocks * bits_per_block
    secret_pixels = capacity_bits // 8  # each pixel is 8 bits
    secret_dim = int(np.sqrt(secret_pixels))  # e.g., for 8192 bits, secret_pixels = 1024, secret_dim = 32

    # Load and resize secret image to secret_dim x secret_dim
    secret_img = Image.open(secret_image_path).convert('L')
    secret_img_resized = secret_img.resize((secret_dim, secret_dim))
    secret_np = np.array(secret_img_resized, dtype=np.uint8)
    # Convert secret image to a binary string (8 bits per pixel)
    secret_bits = ''.join(format(pixel, '08b') for pixel in secret_np.flatten())
    # Check that we have exactly capacity_bits bits
    if len(secret_bits) != capacity_bits:
        raise ValueError(f"Secret bit stream length {len(secret_bits)} does not match capacity {capacity_bits}.")

    # We'll embed bits sequentially from secret_bits
    bit_index = 0
    stego_np = np.empty_like(cover_np)
    
    # Process cover image block-by-block
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            # Extract current block
            row = i * block_size
            col = j * block_size
            block = cover_np[row:row+block_size, col:col+block_size]
            # Compute DCT of the block
            dct_block = dct2(block)
            # Extract next bits to embed (bits_per_block bits)
            bits_to_embed = secret_bits[bit_index: bit_index + bits_per_block]
            bit_index += bits_per_block
            # Modify the chosen DCT coefficient
            orig_coef = dct_block[target_coeff]
            new_coef = embed_bits_in_coefficient(orig_coef, bits_to_embed, num_bits=bits_per_block)
            dct_block[target_coeff] = new_coef
            # Compute inverse DCT to get the stego block
            block_idct = idct2(dct_block)
            stego_np[row:row+block_size, col:col+block_size] = block_idct

    # Clip values to 0-255 and convert to uint8 for saving
    stego_np = np.clip(np.round(stego_np), 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_np, mode='L')
    stego_img.save(stego_image_path)
    print(f"Stego image saved to {stego_image_path}")
    return cover_np.astype(np.uint8), secret_np, stego_np

def extract_secret_image_dct(stego_image_path, bits_per_block=2, target_coeff=(3,3), cover_shape=(512,512)):
    """
    Extract the secret image from a stego image that was embedded in the DCT domain.
    
    The stego image is divided into 8×8 blocks. In each block, the 2 LSBs of the chosen DCT
    coefficient (default (3,3)) are extracted. The concatenated bits form the secret bitstream,
    which is then parsed into bytes to reconstruct the secret image.
    
    Assumes the cover image is 512×512, so there are 4096 blocks and total capacity = 4096*bits_per_block bits.
    The secret image is assumed to be square with dimension = sqrt(total_pixels), where total_pixels = (capacity_bits/8).
    """
    # Load stego image
    stego_img = Image.open(stego_image_path).convert('L')
    stego_np = np.array(stego_img, dtype=np.float32)
    height, width = cover_shape  # should be 512,512
    block_size = 8
    num_blocks_vert = height // block_size
    num_blocks_horiz = width // block_size
    total_blocks = num_blocks_vert * num_blocks_horiz
    capacity_bits = total_blocks * bits_per_block
    secret_pixels = capacity_bits // 8
    secret_dim = int(np.sqrt(secret_pixels))
    
    extracted_bits = ""
    for i in range(num_blocks_vert):
        for j in range(num_blocks_horiz):
            row = i * block_size
            col = j * block_size
            block = stego_np[row:row+block_size, col:col+block_size]
            dct_block = dct2(block)
            # Round the coefficient and convert to int16 for extraction
            coef_val = np.int16(round(dct_block[target_coeff]))
            # Interpret as unsigned 16-bit to extract bits
            coef_uint = np.uint16(coef_val)
            # Extract the lower 'bits_per_block' bits
            bits_val = coef_uint & ((1 << bits_per_block) - 1)
            extracted_bits += format(bits_val, f'0{bits_per_block}b')
    
    # Now, parse the extracted bitstream into bytes (8 bits each)
    secret_data = []
    for i in range(0, len(extracted_bits), 8):
        byte_str = extracted_bits[i:i+8]
        secret_data.append(int(byte_str, 2))
    secret_data = np.array(secret_data, dtype=np.uint8)
    secret_image_np = secret_data.reshape((secret_dim, secret_dim))
    extracted_secret_img = Image.fromarray(secret_image_np, mode='L')
    return secret_image_np, extracted_secret_img

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
    # File paths (update as needed)
    cover_image_path = 'lena.png'      # 512x512 cover image (grayscale)
    secret_image_path = '../../mark.png'    # Secret image (will be resized to 32x32)
    stego_image_path = 'stego_dct.png'
    
    # Embed secret image using DCT
    cover_img_np, secret_img_resized_np, stego_np = embed_secret_image_dct(
        cover_image_path, secret_image_path, stego_image_path, bits_per_block=2, target_coeff=(3,3))
    
    # Extract secret image from stego image
    extracted_secret_np, extracted_secret_img = extract_secret_image_dct(
        stego_image_path, bits_per_block=2, target_coeff=(3,3), cover_shape=cover_img_np.shape)
    
    # Compute MSE/PSNR for cover vs. stego (in spatial domain)
    mse_cover = compute_mse(cover_img_np, stego_np)
    psnr_cover = compute_psnr(mse_cover)
    
    # Compute MSE/PSNR for secret image vs. extracted secret (the secret image was resized to 32x32)
    mse_secret = compute_mse(secret_img_resized_np, extracted_secret_np)
    psnr_secret = compute_psnr(mse_secret)
    
    # Display results using Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top left: Cover Image
    axes[0,0].imshow(cover_img_np, cmap='gray')
    axes[0,0].set_title("Cover Image")
    axes[0,0].axis("off")
    
    # Top right: Stego Image
    axes[0,1].imshow(stego_np, cmap='gray')
    axes[0,1].set_title("Stego Image (DCT Embedding)")
    axes[0,1].axis("off")
    
    # Bottom left: Original Secret Image (resized to 32x32)
    axes[1,0].imshow(secret_img_resized_np, cmap='gray')
    axes[1,0].set_title("Original Secret Image (32x32)")
    axes[1,0].axis("off")
    
    # Bottom right: Extracted Secret Image
    axes[1,1].imshow(extracted_secret_np, cmap='gray')
    axes[1,1].set_title("Extracted Secret Image")
    axes[1,1].axis("off")
    
    fig.suptitle(
        f"Cover vs. Stego: MSE = {mse_cover:.2f}, PSNR = {psnr_cover:.2f} dB\n"
        f"Secret vs. Extracted: MSE = {mse_secret:.2f}, PSNR = {psnr_secret:.2f} dB",
        fontsize=14)
    
    plt.tight_layout()
    plt.show()

