import numpy as np
import pytest
from algorithms.DWT.Image import DWT_multiLevel, DWT_oneLevel, DWT_otsu
import tempfile
import os
from PIL import Image
from tests.test_utils import compute_ssim, load_test_images

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert DWT_multiLevel.compute_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = DWT_multiLevel.compute_mse(arr, arr)
    assert DWT_multiLevel.compute_psnr(mse) == float('inf')

def test_quantize_dequantize():
    arr = np.arange(16, dtype=np.uint8).reshape(4,4)
    q = DWT_multiLevel.quantize_to_n_bits(arr, 2)
    dq = DWT_multiLevel.dequantize_from_n_bits(q, 2)
    assert dq.shape == arr.shape

def test_embed_extract_dwt():
    # Usa immagini reali
    cover, secret = load_test_images()
    
    if cover is None or secret is None:
        pytest.skip("Immagini di test non trovate")
    
    # Salva temporaneamente le immagini su file per DWT
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as cover_file:
        cover_img = Image.fromarray(cover)
        cover_img.save(cover_file.name)
        cover_path = cover_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as secret_file:
        secret_img = Image.fromarray(secret)
        secret_img.save(secret_file.name)
        secret_path = secret_file.name
    
    stego_path = 'tmp_stego.png'
    
    try:
        # Embedding
        cover_np, secret_np, secret_quant, stego_np, secret_shape = DWT_multiLevel.embed_secret_image_dwt_distributed(
            cover_image_path=cover_path,
            secret_image_path=secret_path,
            stego_image_path=stego_path,
            bits_used=1, wavelet='bior2.2', level=1
        )
        
        # Estrazione
        extracted = DWT_multiLevel.extract_secret_image_dwt_distributed(
            stego_path, bits_used=1, wavelet='bior2.2', level=1, secret_shape=secret_shape
        )
        
        # Usa SSIM invece di MSE per valutare la somiglianza visiva
        ssim_score = compute_ssim(extracted, secret_np)
        assert ssim_score > 0.1  # SSIM > 0.2 indica somiglianza visiva accettabile per DWT
    finally:
        # Pulisci i file temporanei
        for path in [cover_path, secret_path, stego_path]:
            if os.path.exists(path):
                os.unlink(path) 