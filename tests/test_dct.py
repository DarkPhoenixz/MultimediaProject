import numpy as np
import pytest
from algorithms.DCT.Image import DCT-full, DCT-JPEG-like, DCT-lowFreq, DCT-singleQuant
from algorithms.DCT.Text import DCT-text

# --- Test utilità comuni ---
def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert DCT-full.compute_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = DCT-full.compute_mse(arr, arr)
    assert DCT-full.compute_psnr(mse) == float('inf')

def test_mse_simple():
    arr1 = np.zeros((5, 5), dtype=np.uint8)
    arr2 = np.ones((5, 5), dtype=np.uint8) * 10
    assert DCT-full.compute_mse(arr1, arr2) == 100

# --- Test embedding/estrazione DCT full ---
def test_embed_extract_dct_full():
    cover = np.ones((64, 64), dtype=np.uint8) * 120
    watermark = np.ones((64, 64), dtype=np.uint8) * 30
    alpha = 1
    _, _, watermarked, _ = DCT-full.embed_watermark_dct(cover, watermark, alpha)
    extracted = DCT-full.extract_watermark_dct(watermarked, cover, alpha)
    # L'estratto deve essere simile al watermark
    assert np.mean(np.abs(extracted - watermark)) < 10

# --- Test embedding/estrazione DCT-text ---
def test_embed_extract_dct_text():
    cover = np.ones((32, 32), dtype=np.uint8) * 100
    secret = "testmessage"
    block_size = 8
    quant_matrix = np.ones((8,8), dtype=np.float32)
    target_coeff = (3,3)
    bits_per_block = 1
    stego_np, _ = DCT-text.embed_secret_dct_string(cover, secret, block_size, quant_matrix, target_coeff, bits_per_block)
    extracted = DCT-text.extract_secret_dct_string(stego_np, block_size, quant_matrix, target_coeff, bits_per_block)
    assert secret in extracted 