import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

# Convert a message string to a binary string
def message_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

# Convert a binary string back to a message string
def binary_to_message(binary):
    message = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:  # Ensure full byte
            message.append(chr(int(byte, 2)))
    return ''.join(message)

# Apply Discrete Cosine Transform (DCT) to an 8x8 block
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Apply Inverse Discrete Cosine Transform (IDCT) to an 8x8 block
def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Embed a secret message into the DCT coefficients of an image
def embed_message(image_path, secret_message, output_path):
    # Load the image and convert it to YCbCr format
    image = Image.open(image_path).convert('YCbCr')
    y, cb, cr = image.split()
    y_data = np.array(y, dtype=np.float32)
    
    # Get the image dimensions and pad to multiples of 8
    height, width = y_data.shape
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    y_data_padded = np.pad(y_data, ((0, padded_height - height), (0, padded_width - width)), mode='constant')
    
    # Convert the message to binary
    binary_message = message_to_binary(secret_message)
    message_index = 0
    
    # Process each 8x8 block
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            block = y_data_padded[i:i+8, j:j+8]
            dct_block = apply_dct(block)
            
            # Embed the message into the LSB of DCT coefficients
            for k in range(8):
                for m in range(8):
                    if message_index < len(binary_message):
                        dct_block[k, m] = int(dct_block[k, m]) & ~1 | int(binary_message[message_index])
                        message_index += 1
            
            # Apply inverse DCT
            idct_block = apply_idct(dct_block)
            y_data_padded[i:i+8, j:j+8] = idct_block
    
    # Crop back to the original dimensions and convert to an image
    y_data = y_data_padded[:height, :width]
    y_data = np.clip(y_data, 0, 255).astype(np.uint8)
    y_image = Image.fromarray(y_data, mode='L')
    
    # Merge the YCbCr channels and save the result
    result_image = Image.merge('YCbCr', (y_image, cb, cr))
    result_image.save(output_path)

# Extract the secret message from the DCT coefficients of an image
def extract_message(image_path):
    image = Image.open(image_path).convert('YCbCr')
    y, cb, cr = image.split()
    y_data = np.array(y, dtype=np.float32)
    
    height, width = y_data.shape
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    y_data_padded = np.pad(y_data, ((0, padded_height - height), (0, padded_width - width)), mode='constant')
    
    binary_message = []
    
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            block = y_data_padded[i:i+8, j:j+8]
            dct_block = apply_dct(block)
            
            # Extract the LSBs from DCT coefficients
            for k in range(8):
                for m in range(8):
                    binary_message.append(str(int(dct_block[k, m]) % 2))
    
    # Convert the binary string back to the message
    binary_message = ''.join(binary_message)
    message = binary_to_message(binary_message)
    return message

# Display the original and steganographic images using matplotlib
def display_images(original_path, stego_path):
    original_image = Image.open(original_path).convert('RGB')
    stego_image = Image.open(stego_path).convert('RGB')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Steganographic Image")
    plt.imshow(stego_image)
    plt.axis('off')
    
    plt.show()

# Test code
if __name__ == "__main__":
        
    input_image = "../../lena.png"  # Path to the original image
    output_image = "output.jpg"  # Path to save the steganographic image
    secret_message = "This is a hidden message!"  # Secret message to embed
    
    # Embed the message
    embed_message(input_image, secret_message, output_image)
    print("Message successfully embedded into the image.")
    
    # Extract the message
    extracted_message = extract_message(output_image)
    print("Extracted message:", extracted_message)
    
    # Display the original and steganographic images
    display_images(input_image, output_image)