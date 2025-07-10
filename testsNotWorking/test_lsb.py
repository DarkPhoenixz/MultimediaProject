import numpy as np
import pytest
from algorithms.LSB.Image import LSB_spatial
import os
from PIL import Image
from tests.test_utils import compute_ssim, load_test_images

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert LSB_spatial.compute_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = LSB_spatial.compute_mse(arr, arr)
    assert LSB_spatial.compute_psnr(mse) == float('inf')

def test_quantize_dequantize():
    arr = np.arange(16, dtype=np.uint8).reshape(4,4)
    q = LSB_spatial.quantize_to_n_bits(arr, 2)
    dq = LSB_spatial.dequantize_from_n_bits(q, 2)
    assert dq.shape == arr.shape

def test_embed_extract_lsb():
    # Usa immagini reali
    cover, secret = load_test_images()
    
    if cover is None or secret is None:
        pytest.skip("Immagini di test non trovate")
    
    bits_used = 1
    
    # embed_secret_image restituisce (cover_np, secret_np, secret_quant, stego_np)
    result = LSB_spatial.embed_secret_image(cover, secret, bits_used)
    stego_np = result[3]  # stego_np è il quarto elemento della tupla
    
    extracted = LSB_spatial.extract_secret_image(stego_np, bits_used)
    
    # Usa SSIM invece di MSE per valutare la somiglianza visiva
    ssim_score = compute_ssim(extracted, secret)
    assert ssim_score > 0.4  # SSIM > 0.4 indica somiglianza visiva accettabile per LSB 