import numpy as np
import pytest
from algorithms.DFT.Image import DFT

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
    cover = np.ones((64, 64), dtype=np.uint8) * 120
    watermark = np.ones((32, 32), dtype=np.uint8) * 30
    alpha = 1
    # Simula embedding
    dft_cover = DFT.apply_dft(cover)
    dft_watermarked = DFT.embed_watermark(dft_cover, watermark, alpha)
    # Simula estrazione
    extracted = DFT.extract_watermark(dft_cover, dft_watermarked, alpha)
    # L'estratto deve essere simile al watermark (dopo normalizzazione)
    assert np.mean(np.abs(extracted - watermark)) < 10 