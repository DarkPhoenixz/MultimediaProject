import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10(255**2 / mse)
    return psnr

def extract_watermark(original_dft, watermarked_dft, alpha):
    magnitude_original = np.abs(original_dft)
    magnitude_watermarked = np.abs(watermarked_dft)
    extracted_watermark = (magnitude_watermarked - magnitude_original) / alpha
    return extracted_watermark

def normalize_image(image):
    # Clip values to avoid division by zero or log of negative values
    image = np.clip(image, -1e-9, None)
    
    # Use logarithmic scaling to compress the dynamic range
    log_image = np.log1p(image + 1e-9)
    
    # Scale the image to the range [0, 255]
    min_val = np.min(log_image)
    max_val = np.max(log_image)
    normalized_image = (log_image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)

def display_results(original, watermarked, original_dft, watermarked_dft, extracted_watermark, psnr_value):
    plt.figure(figsize=(12, 8))

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
    magnitude_spectrum = 20 * np.log(np.abs(original_dft) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Watermarked DFT Spectrum")
    magnitude_spectrum = 20 * np.log(np.abs(watermarked_dft) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    print(f"Extracted Watermark min: {np.min(extracted_watermark)}, max: {np.max(extracted_watermark)}")

    # Normalize the extracted watermark for matplotlib
    extracted_watermark_normalized = normalize_image(extracted_watermark)
    plt.subplot(2, 3, 5)
    plt.title("Extracted Watermark")
    plt.imshow(extracted_watermark_normalized, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, f"PSNR: {psnr_value:.2f} dB", fontsize=16, ha='center', va='center')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    lena_image = load_image_grayscale("../../lena.png")
    logo_image = load_image_grayscale("../../mark.png")
    logo_resized = resize_watermark(logo_image, lena_image.shape)
    dft_shifted = apply_dft(lena_image)

    alpha = 0.1  # Increase watermark intensity
    watermarked_dft = embed_watermark(dft_shifted, logo_resized, alpha)
    watermarked_image = inverse_dft(watermarked_dft)
    watermarked_image = cv2.convertScaleAbs(watermarked_image, alpha=(255.0 / np.max(watermarked_image)))

    psnr_value = calculate_psnr(lena_image, watermarked_image)
    extracted_watermark = extract_watermark(dft_shifted, watermarked_dft, alpha)

    display_results(lena_image, watermarked_image, dft_shifted, watermarked_dft, extracted_watermark, psnr_value)

if __name__ == "__main__":
    main()