import numpy as np
import pytest
from algorithms.DSSS.Image import DSSS
import tempfile
import os
import cv2
from tests.test_utils import compute_ssim, load_test_images

def test_mse_identical():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    assert DSSS.mse(arr, arr) == 0

def test_psnr_inf():
    arr = np.ones((10, 10), dtype=np.uint8) * 100
    mse = DSSS.mse(arr, arr)
    assert DSSS.psnr(arr, arr) == float('inf')

def test_mse_simple():
    arr1 = np.zeros((5, 5), dtype=np.uint8)
    arr2 = np.ones((5, 5), dtype=np.uint8) * 10
    assert DSSS.mse(arr1, arr2) == 100

def test_pseudo_random_sequence():
    seq1 = DSSS.generate_pseudo_random_sequence(42, (8,8))
    seq2 = DSSS.generate_pseudo_random_sequence(42, (8,8))
    assert np.array_equal(seq1, seq2)
    seq3 = DSSS.generate_pseudo_random_sequence(43, (8,8))
    assert not np.array_equal(seq1, seq3)

def test_embed_extract_dsss():
    # Usa immagini reali
    cover, watermark = load_test_images()
    
    if cover is None or watermark is None:
        pytest.skip("Immagini di test non trovate")
    
    # Salva temporaneamente le immagini su file per DSSS
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as cover_file:
        cv2.imwrite(cover_file.name, cover)
        cover_path = cover_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as watermark_file:
        cv2.imwrite(watermark_file.name, watermark)
        watermark_path = watermark_file.name
    
    try:
        alpha = 1
        seed = 42
        
        # Simula embedding
        original, watermark_binary, watermarked = DSSS.embed_watermark(cover_path, watermark_path, alpha, seed)
        # Simula estrazione
        extracted = DSSS.extract_watermark(watermarked, original, alpha, seed)
        
        # Usa SSIM invece di MSE per valutare la somiglianza visiva
        watermark_uint8 = (watermark_binary * 255).astype(np.uint8)
        ssim_score = compute_ssim(extracted, watermark_uint8)
        assert ssim_score > 0.1  # SSIM > 0.1 indica somiglianza visiva accettabile per DSSS
    finally:
        # Pulisci i file temporanei
        for path in [cover_path, watermark_path]:
            if os.path.exists(path):
                os.unlink(path) 