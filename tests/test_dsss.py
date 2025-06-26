import numpy as np
import pytest
from algorithms.DSSS.Image import DSSS

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
    cover = np.ones((32, 32), dtype=np.uint8) * 120
    watermark = np.ones((32, 32), dtype=np.uint8) * 30
    alpha = 1
    seed = 42
    # Simula embedding
    watermarked, _ = DSSS.embed_watermark(cover, watermark, alpha, seed)
    # Simula estrazione
    extracted = DSSS.extract_watermark(watermarked, cover, alpha, seed)
    assert np.mean(np.abs(extracted - watermark)) < 10 