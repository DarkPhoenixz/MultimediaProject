import numpy as np
import pytest
from algorithms.LSB.Image import LSB-spatial

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert LSB-spatial.compute_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = LSB-spatial.compute_mse(arr, arr)
    assert LSB-spatial.compute_psnr(mse) == float('inf')

def test_quantize_dequantize():
    arr = np.arange(16, dtype=np.uint8).reshape(4,4)
    q = LSB-spatial.quantize_to_n_bits(arr, 2)
    dq = LSB-spatial.dequantize_from_n_bits(q, 2)
    assert dq.shape == arr.shape

def test_embed_extract_lsb():
    cover = np.ones((32, 32), dtype=np.uint8) * 120
    secret = np.ones((32, 32), dtype=np.uint8) * 30
    bits_used = 1
    stego = LSB-spatial.embed_secret_image(cover, secret, bits_used)
    extracted = LSB-spatial.extract_secret_image(stego, bits_used)
    assert np.mean(np.abs(extracted - secret)) < 10 