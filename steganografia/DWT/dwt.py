from PIL import Image
import numpy as np

def embed_message_spatial(image_path, secret_message, output_path):
    """
    Embed a secret message into an image by modifying the least significant bits
    of its pixel values (spatial domain embedding).
    
    The first 32 bits of the embedded data store the length (in bits) of the secret message.
    """
    # Load the image as grayscale (8-bit)
    image = Image.open(image_path).convert('L')
    data = np.array(image, dtype=np.uint8)
    flat_data = data.flatten()

    # Convert the secret message into a binary string (8 bits per character)
    binary_message = ''.join(format(ord(char), '08b') for char in secret_message)
    message_length = len(binary_message)
    
    # Embed the message length (32 bits) at the beginning
    length_bin = format(message_length, '032b')
    full_message = length_bin + binary_message

    # Check if the message can fit in the image
    if len(full_message) > flat_data.size:
        raise ValueError("Message is too long for the provided image.")

    # Modify the LSB of each pixel for the bits we need to embed.
    # Use 0xFE as the mask to clear the LSB.
    for i, bit in enumerate(full_message):
        flat_data[i] = (flat_data[i] & 0xFE) | int(bit)

    # Reshape to the original image shape and save
    new_data = flat_data.reshape(data.shape)
    new_image = Image.fromarray(new_data, mode='L')
    new_image.save(output_path)
    print(f"Message embedded successfully. Image saved to {output_path}")

def extract_message_spatial(image_path):
    """
    Extract the secret message from an image that has been embedded by modifying pixel LSBs.
    
    Reads the first 32 bits to get the length (in bits) of the message, then extracts
    that many bits and converts them back to text.
    """
    # Load the image as grayscale (8-bit)
    image = Image.open(image_path).convert('L')
    data = np.array(image, dtype=np.uint8)
    flat_data = data.flatten()

    # Extract the LSB from every pixel and combine them into a string
    extracted_bits = ''.join(str(pixel & 1) for pixel in flat_data)

    # The first 32 bits encode the length of the secret message (in bits)
    length_bits = extracted_bits[:32]
    message_length = int(length_bits, 2)
    print(f"Extracted message bit length: {message_length}")

    # Extract the secret message bits
    message_bits = extracted_bits[32:32 + message_length]

    # Convert each 8-bit segment into a character
    secret_message = ''.join(
        chr(int(message_bits[i:i+8], 2))
        for i in range(0, len(message_bits), 8)
        if len(message_bits[i:i+8]) == 8
    )
    print(f"Extracted message: {secret_message}")

# Example usage:
if __name__ == "__main__":
    input_image_path = 'input.png'   # Path to your input image (PNG)
    secret_message = 'Hello, this is a secret messageeeeeeeee!'  # Your secret message
    output_image_path = 'output.png'  # Path where the output image will be saved

    try:
        embed_message_spatial(input_image_path, secret_message, output_image_path)
    except ValueError as e:
        print(e)
        exit()

    extract_message_spatial(output_image_path)
