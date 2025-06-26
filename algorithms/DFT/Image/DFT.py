import timeit
import cv2
import numpy as np
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

def load_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")
    return image

def apply_dft(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def inverse_dft(dft_shifted):
    f_ishift = np.fft.ifftshift(dft_shifted)
    image_back = np.fft.ifft2(f_ishift)
    return np.abs(image_back)

def embed_watermark(dft_shifted, watermark, alpha):
    magnitude = np.abs(dft_shifted)
    phase = np.angle(dft_shifted)
    
    # Normalize the watermark to avoid distortions
    mean_val = np.mean(watermark)
    watermark = watermark - mean_val
    watermark = watermark / np.max(np.abs(watermark))

    # Add the watermark to the magnitude
    watermarked_magnitude = magnitude * (1 + alpha * watermark)

    # Reconstruct the DFT with the modified magnitude
    watermarked_dft = watermarked_magnitude * np.exp(1j * phase)
    return watermarked_dft

def resize_watermark(watermark, target_shape):
    return cv2.resize(watermark, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

def calculate_psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10(255**2 / mse)
    return psnr

def calculate_mse(original, modified):
    mse = np.mean((original - modified) ** 2)
    return mse

def extract_watermark(original_dft, watermarked_dft, alpha):
    magnitude_original = np.abs(original_dft)
    magnitude_watermarked = np.abs(watermarked_dft)
    extracted_watermark = (magnitude_watermarked - magnitude_original) / alpha
    return extracted_watermark

def normalize_image(image):
    # Clip values to avoid negative or zero for log operation
    image = np.clip(image, -1e-9, None)
    # Logarithmic scaling to compress dynamic range
    log_image = np.log1p(image + 1e-9)
    # Scale to [0, 255]
    min_val = np.min(log_image)
    max_val = np.max(log_image)
    normalized_image = (log_image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)

def display_results(original, watermarked, original_dft, watermarked_dft, 
                    extracted_watermark, psnr_main, mse_main, watermark_original, embedTime, extractTime):
    plt.figure(figsize=(12, 8))
    
    # Compute magnitude spectra for display
    spectrum_original = 20 * np.log(np.abs(original_dft) + 1)
    spectrum_watermarked = 20 * np.log(np.abs(watermarked_dft) + 1)
    
    # Normalize extracted watermark for display
    extracted_disp = normalize_image(extracted_watermark)
    
    # --- Compute Watermark Metrics ---
    # Normalize the original watermark (used for embedding)
    wm_orig_norm = watermark_original.astype(np.float32)
    wm_orig_norm = wm_orig_norm - np.mean(wm_orig_norm)
    wm_orig_norm = wm_orig_norm / np.max(np.abs(wm_orig_norm))
    
    # Normalize the extracted watermark in the same way
    extracted_norm = extracted_watermark.copy()
    extracted_norm = extracted_norm - np.mean(extracted_norm)
    extracted_norm = extracted_norm / np.max(np.abs(extracted_norm))
    
    mse_wm = np.mean((wm_orig_norm - extracted_norm)**2)
    # Assuming watermark is normalized to [-1,1] (max value=1)
    psnr_wm = 10 * np.log10(1.0 / mse_wm) if mse_wm != 0 else float("inf")
    
    # --- Display using subplots ---
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Watermarked Image")
    plt.imshow(watermarked, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Original DFT Spectrum")
    plt.imshow(spectrum_original, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Watermarked DFT Spectrum")
    plt.imshow(spectrum_watermarked, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("Extracted Watermark")
    plt.imshow(extracted_disp, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.7, f"Embedding Time: {embedTime:.2f} s\nExtraction Time: {extractTime:.2f} s",
             fontsize=14, ha='center', va='center')
    plt.text(0.5, 0.5, f"Main Image PSNR: {psnr_main:.2f} dB\nMain Image MSE: {mse_main:.2f}", 
             fontsize=14, ha='center', va='center')
    plt.text(0.5, 0.3, f"Watermark PSNR: {psnr_wm:.2f} dB\nWatermark MSE: {mse_wm:.2e}", 
             fontsize=14, ha='center', va='center')
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    # Load main image and watermark
    lena_image = load_image_grayscale(cover_image_path)
    logo_image = load_image_grayscale(secret_image_path)
    
    # Resize watermark to match main image dimensions
    logo_resized = resize_watermark(logo_image, lena_image.shape)
    
    # Compute DFT of main image
    dft_shifted = apply_dft(lena_image)
    
    # Embed watermark with chosen intensity
    alpha = 0.1  # Watermark intensity
    watermarked_dft = embed_watermark(dft_shifted, logo_resized, alpha)
    
    # Reconstruct watermarked image via inverse DFT
    watermarked_image = inverse_dft(watermarked_dft)
    watermarked_image = cv2.convertScaleAbs(watermarked_image, alpha=(255.0 / np.max(watermarked_image)))
    
    # Calculate PSNR and MSE for main image
    psnr_main = calculate_psnr(lena_image, watermarked_image)
    mse_main = calculate_mse(lena_image, watermarked_image)
    print(f"Main Image PSNR: {psnr_main:.2f} dB")
    print(f"Main Image MSE: {mse_main:.2f}")
    
    # Extract watermark from DFT
    extracted_watermark = extract_watermark(dft_shifted, watermarked_dft, alpha)
    
    # Calculate embedding computational time
    def embed_wrapper():
        embed_watermark(dft_shifted, logo_resized, alpha)
    embedTime = timeit.timeit(embed_wrapper, number=10)     

    # Calculate extraction computational time
    def extract_wrapper():
        extract_watermark(dft_shifted, watermarked_dft, alpha)
    extractTime = timeit.timeit(extract_wrapper, number=10)

    # Display results (pass the original watermark used for embedding)
    display_results(lena_image, watermarked_image, dft_shifted, watermarked_dft, 
                    extracted_watermark, psnr_main, mse_main, logo_resized, embedTime, extractTime)
    
    # Save watermarked image
    cv2.imwrite("watermarked_lena.png", watermarked_image)

if __name__ == "__main__":
    main()

#non ci sono osservazione al momento
#funziona