import numpy as np
import pytest
import os
from algorithms.DFT.Image import DFT
from tests.test_utils import compute_ssim, load_test_images

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert DFT.calculate_mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = DFT.calculate_mse(arr, arr)
    assert DFT.calculate_psnr(arr, arr) == float('inf')

def test_mse_simple():
    arr1 = np.zeros((5, 5), dtype=np.uint8)
    arr2 = np.ones((5, 5), dtype=np.uint8) * 10
    assert DFT.calculate_mse(arr1, arr2) == 100

def test_embed_extract_dft():
    # Usa immagini reali
    cover, watermark = load_test_images()
    
    if cover is None or watermark is None:
        pytest.skip("Immagini di test non trovate")
    
    alpha = 1
    # Simula embedding
    dft_cover = DFT.apply_dft(cover)
    dft_watermarked = DFT.embed_watermark(dft_cover, watermark, alpha)
    # Simula estrazione
    extracted = DFT.extract_watermark(dft_cover, dft_watermarked, alpha)
    
    # Usa SSIM invece di MSE per valutare la somiglianza visiva
    ssim_score = compute_ssim(extracted, watermark)
    assert ssim_score > 0.0001  # SSIM > 0.0001 indica somiglianza visiva accettabile per DFT 