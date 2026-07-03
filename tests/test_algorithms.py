"""
test_algorithms.py
Berisi unit tests menggunakan pytest untuk memvalidasi algoritma 
sekuensial dan kasus-kasus batas.
"""
import pytest
import numpy as np
import sys
import os

# Memastikan modul di src bisa diimpor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from common import generate_matrix, get_test_matrix, calculate_checksum
from sequential import matrix_multiply_sequential, find_row_max_sequential

def test_3x3_matrix_multiplication():
    """Pengujian perkalian matriks 3x3 sesuai hasil manual."""
    matrix = get_test_matrix()
    result = matrix_multiply_sequential(matrix)
    expected_result = np.array([
        [30, 36, 42],
        [66, 81, 96],
        [102, 126, 150]
    ], dtype=np.int64)
    np.testing.assert_array_equal(result, expected_result)

def test_find_row_max():
    """Pengujian nilai maksimum setiap baris bernilai benar."""
    matrix = get_test_matrix()
    result = matrix_multiply_sequential(matrix)
    max_vals, max_cols = find_row_max_sequential(result)
    expected_max_vals = np.array([42, 96, 150], dtype=np.int64)
    expected_max_cols = np.array([2, 2, 2], dtype=np.int32)
    
    np.testing.assert_array_equal(max_vals, expected_max_vals)
    np.testing.assert_array_equal(max_cols, expected_max_cols)

def test_same_max_picks_first_index():
    """Jika ada dua maksimum sama, indeks pertama yang dipilih."""
    # Matriks custom dimana ada hasil nilai sama
    dummy_result = np.array([
        [10, 50, 50, 20],
        [100, 100, 50, 10]
    ], dtype=np.int64)
    
    max_vals, max_cols = find_row_max_sequential(dummy_result)
    
    # Baris 0: maks 50 di col 1 dan 2. Harus pilih 1.
    # Baris 1: maks 100 di col 0 dan 1. Harus pilih 0.
    np.testing.assert_array_equal(max_vals, np.array([50, 100]))
    np.testing.assert_array_equal(max_cols, np.array([1, 0]))

def test_checksum_consistency():
    """Checksum konsisten untuk input yang sama."""
    matrix = get_test_matrix()
    result = matrix_multiply_sequential(matrix)
    checksum1 = calculate_checksum(result)
    checksum2 = calculate_checksum(result)
    assert checksum1 == checksum2
    assert checksum1 == 729

def test_zero_size_input_rejected():
    """Input ukuran nol ditolak saat pembuatan matriks."""
    with pytest.raises(ValueError):
        generate_matrix(0, 123)

def test_negative_size_input_rejected():
    """Input ukuran negatif ditolak saat pembuatan matriks."""
    with pytest.raises(ValueError):
        generate_matrix(-5, 123)
