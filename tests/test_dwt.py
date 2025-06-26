import numpy as np
import pytest
from algorithms.DWT.Image import DWT-multiLevel, DWT-oneLevel, DWT-otsu

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert DWT-multiLevel.compute_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = DWT-multiLevel.compute_mse(arr, arr)
    assert DWT-multiLevel.compute_psnr(mse) == float('inf')

def test_quantize_dequantize():
    arr = np.arange(16, dtype=np.uint8).reshape(4,4)
    q = DWT-multiLevel.quantize_to_n_bits(arr, 2)
    dq = DWT-multiLevel.dequantize_from_n_bits(q, 2)
    assert dq.shape == arr.shape

def test_embed_extract_dwt():
    cover = np.ones((64, 64), dtype=np.uint8) * 120
    secret = np.ones((32, 32), dtype=np.uint8) * 30
    stego_path = 'tmp_stego.png'
    # Embedding
    DWT-multiLevel.embed_secret_image_dwt_distributed(
        cover_image_path=None,  # Funzione va adattata per array
        secret_image_path=None,
        stego_image_path=stego_path,
        bits_used=1, wavelet='bior2.2', level=1
    )
    # Qui si dovrebbe testare estrazione, ma serve refactoring per lavorare solo in memoria 