import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image_grayscale(image_path):
    """Carica un'immagine in scala di grigi"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Errore: impossibile caricare l'immagine {image_path}")
    return image

def apply_dct(image):
    """Applica la DCT all'immagine"""
    return cv2.dct(np.float32(image))

def inverse_dct(dct_image):
    """Applica la DCT inversa"""
    return cv2.idct(dct_image)

def permute_coefficients(dct_coeffs, seed=42):
    """Permuta i coefficienti DCT in modo casuale usando un seed fisso"""
    np.random.seed(seed)
    flat_coeffs = dct_coeffs.flatten()
    permuted_indices = np.random.permutation(len(flat_coeffs))
    return flat_coeffs[permuted_indices], permuted_indices

def inverse_permutation(permuted_coeffs, indices):
    """Inverte la permutazione per ripristinare l'ordine originale"""
    restored_coeffs = np.zeros_like(permuted_coeffs)
    restored_coeffs[indices] = permuted_coeffs
    return restored_coeffs.reshape((512, 512))  # Assumiamo immagini 512x512

def embed_watermark(image, watermark, alpha=10, seed=42):
    """Incorpora il watermark nei coefficienti DCT usando F5"""
    watermark_resized = cv2.resize(watermark, (image.shape[1] // 8, image.shape[0] // 8), interpolation=cv2.INTER_AREA)
    watermark_resized = watermark_resized.astype(np.float32) / 255  # Normalize

    dct_image = apply_dct(image)
    permuted_coeffs, permuted_indices = permute_coefficients(dct_image, seed)

    bit_index = 0
    for i in range(len(permuted_coeffs)):
        if bit_index >= watermark_resized.size:
            break
        if abs(permuted_coeffs[i]) > 10:  # Avoid modifying low-magnitude coefficients
            bit_value = watermark_resized.flatten()[bit_index]
            permuted_coeffs[i] *= (1 + alpha * (2 * bit_value - 1))  # Scale instead of adding directly
            bit_index += 1

    dct_image_modified = inverse_permutation(permuted_coeffs, permuted_indices)
    watermarked_image = inverse_dct(dct_image_modified)

    return watermarked_image, dct_image_modified, permuted_indices, watermark_resized.shape

def extract_watermark(watermarked_image, original_dct, watermark_shape, alpha=10, seed=42):
    """Extract the grayscale watermark from the steganographic image using F5 algorithm."""
    extracted_values = []
    expected_bits = watermark_shape[0] * watermark_shape[1]

    # Apply DCT to the watermarked image
    dct_watermarked = apply_dct(watermarked_image)

    # Apply the same permutation to the DCT coefficients
    permuted_coeffs_watermarked, permuted_indices = permute_coefficients(dct_watermarked, seed)

    # Extract watermark values
    bit_index = 0
    for i in range(len(permuted_coeffs_watermarked)):
        if bit_index >= expected_bits:
            break
        if abs(permuted_coeffs_watermarked[i]) > 0.1:
            diff = permuted_coeffs_watermarked[i] - original_dct.flatten()[permuted_indices[i]]
            value = np.clip(128 + (diff / alpha) * 128, 0, 255)  # Scale values to grayscale range
            extracted_values.append(value)
            bit_index += 1

    # Ensure correct shape and convert to 8-bit image
    extracted_values = np.array(extracted_values[:expected_bits], dtype=np.uint8)

    if len(extracted_values) != expected_bits:
        raise ValueError(f"Extracted watermark size {len(extracted_values)} does not match expected {expected_bits}")

    # Reshape the extracted values into a grayscale image
    extracted_watermark = extracted_values.reshape(watermark_shape)

    # Enhance contrast
    extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX)

    return extracted_watermark.astype(np.uint8)



def display_results(original, watermarked, extracted_watermark, original_watermark):
    """Mostra i risultati dell'algoritmo"""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Watermarked Image")
    plt.imshow(watermarked, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Extracted Watermark")
    plt.imshow(extracted_watermark, cmap="gray")
    plt.axis("off")

    plt.suptitle("F5 Watermarking Results")
    plt.show()

# =======================
#         MAIN
# =======================
def main():
    original_image = load_image_grayscale("lena.jpg")  # Cambia con la tua immagine
    watermark = load_image_grayscale("mark.jpg")  # Cambia con il tuo watermark

    watermarked_image, modified_dct, indices, watermark_shape = embed_watermark(original_image, watermark, alpha=0.1)

    extracted_watermark = extract_watermark(watermarked_image, modified_dct, watermark_shape, alpha=0.1)

    display_results(original_image, watermarked_image, extracted_watermark, watermark)

if __name__ == "__main__":
    main()
