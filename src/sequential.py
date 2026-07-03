"""
sequential.py
Implementasi algoritma perkalian matriks dan pencarian maksimum
secara sekuensial menggunakan Numba njit (tanpa parallel).
"""
import argparse
import time
import numpy as np
from numba import njit
from common import (
    generate_matrix, get_test_matrix, print_matrix_if_small, 
    calculate_checksum, format_result_summary, save_result_to_csv
)
import os

@njit
def matrix_multiply_sequential(matrix: np.ndarray) -> np.ndarray:
    """Melakukan perkalian matriks secara sekuensial dengan 3 nested loop."""
    n = matrix.shape[0]
    result = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += matrix[i, k] * matrix[k, j]
    return result

@njit
def find_row_max_sequential(result_matrix: np.ndarray):
    """Mencari nilai maksimum dan indeks kolomnya untuk setiap baris."""
    n = result_matrix.shape[0]
    max_values = np.zeros(n, dtype=np.int64)
    max_indices = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        current_max = result_matrix[i, 0]
        current_col = 0
        for j in range(1, n):
            if result_matrix[i, j] > current_max:
                current_max = result_matrix[i, j]
                current_col = j
        max_values[i] = current_max
        max_indices[i] = current_col
        
    return max_values, max_indices

def warmup_numba():
    """Melakukan warm-up agar kompilasi JIT tidak masuk waktu pengukuran."""
    dummy_matrix = np.ones((10, 10), dtype=np.int32)
    res = matrix_multiply_sequential(dummy_matrix)
    find_row_max_sequential(res)

def main():
    parser = argparse.ArgumentParser(description="Sequential Matrix Multiplication and Max Search")
    parser.add_argument("--size", type=int, default=512, help="Ukuran matriks N")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--test-mode", action="store_true", help="Gunakan matriks 3x3 untuk pengujian")
    parser.add_argument("--run-idx", type=int, default=1, help="Indeks run untuk iterasi benchmark")
    
    args = parser.parse_args()
    
    # Validasi input
    if args.size <= 0:
        print("Error: Ukuran matriks harus lebih besar dari 0.")
        return
        
    # Warm-up JIT
    warmup_numba()
    
    if args.test_mode:
        print("Menjalankan mode pengujian (Test Mode)...")
        matrix = get_test_matrix()
        args.size = 3
    else:
        matrix = generate_matrix(args.size, args.seed)
        
    # Mulai pengukuran waktu perkalian
    t0_mult = time.perf_counter()
    result_matrix = matrix_multiply_sequential(matrix)
    t1_mult = time.perf_counter()
    mult_time = t1_mult - t0_mult
    
    # Mulai pengukuran waktu pencarian maksimum
    t0_max = time.perf_counter()
    max_values, max_indices = find_row_max_sequential(result_matrix)
    t1_max = time.perf_counter()
    max_time = t1_max - t0_max
    
    total_time = mult_time + max_time
    checksum = calculate_checksum(result_matrix)
    
    # Validasi
    is_valid = True
    if args.test_mode:
        expected_result = np.array([
            [30, 36, 42],
            [66, 81, 96],
            [102, 126, 150]
        ], dtype=np.int64)
        expected_max_values = np.array([42, 96, 150], dtype=np.int64)
        expected_max_indices = np.array([2, 2, 2], dtype=np.int32)
        
        is_valid = (np.array_equal(result_matrix, expected_result) and 
                    np.array_equal(max_values, expected_max_values) and 
                    np.array_equal(max_indices, expected_max_indices))
    else:
        # Untuk mode benchmark, simpan referensi hasil sekuensial
        os.makedirs('results', exist_ok=True)
        ref_file = f"results/reference_n{args.size}_seed{args.seed}.npz"
        np.savez(ref_file, 
                 result_matrix=result_matrix, 
                 max_values=max_values, 
                 max_indices=max_indices, 
                 checksum=checksum,
                 n=args.size,
                 seed=args.seed)
    
    # Menampilkan hasil
    print(format_result_summary(
        method="Sequential Numba",
        n=args.size,
        seed=args.seed,
        workers=1,
        mult_time=mult_time,
        max_time=max_time,
        total_time=total_time,
        checksum=checksum,
        is_valid=is_valid
    ))
    
    print("\nLima hasil baris pertama (maksimum dan indeks kolom):")
    limit = min(5, args.size)
    for i in range(limit):
        print(f"Row {i}: max = {max_values[i]}, column = {max_indices[i]}")
        
    if not args.test_mode:
        save_result_to_csv(
            filepath="results/benchmark.csv",
            method="sequential",
            n=args.size,
            workers=1,
            run_idx=args.run_idx,
            mult_time=mult_time,
            max_time=max_time,
            total_time=total_time,
            checksum=checksum,
            is_valid=is_valid
        )

if __name__ == "__main__":
    main()
