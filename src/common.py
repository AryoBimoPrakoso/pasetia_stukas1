"""
common.py
Berisi fungsi-fungsi utilitas untuk pembuatan data, perhitungan checksum,
dan pencetakan matriks serta hasil.
"""
import os
import numpy as np

def generate_matrix(n: int, seed: int) -> np.ndarray:
    """Membuat matriks N x N dengan nilai acak 0-9 menggunakan seed tertentu."""
    if n <= 0:
        raise ValueError("Ukuran matriks harus bilangan bulat positif.")
    np.random.seed(seed)
    return np.random.randint(0, 10, size=(n, n), dtype=np.int32)

def get_test_matrix() -> np.ndarray:
    """Mengembalikan matriks 3x3 tetap untuk mode pengujian."""
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.int32)

def print_matrix_if_small(matrix: np.ndarray, max_size: int = 10) -> None:
    """Mencetak matriks jika ukurannya tidak lebih dari max_size."""
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        print(matrix)

def calculate_checksum(matrix: np.ndarray) -> int:
    """Menghitung checksum dari sebuah matriks (jumlah seluruh elemen)."""
    # Gunakan np.int64 untuk mencegah overflow saat menjumlahkan matriks besar
    return int(np.sum(matrix, dtype=np.int64))

def compare_results(result_a: np.ndarray, result_b: np.ndarray) -> bool:
    """Membandingkan dua matriks untuk memastikan hasilnya identik."""
    return np.array_equal(result_a, result_b)

def compare_maximum_results(max_vals_a: np.ndarray, max_cols_a: np.ndarray, 
                            max_vals_b: np.ndarray, max_cols_b: np.ndarray) -> bool:
    """Membandingkan array nilai maksimum dan indeks kolomnya."""
    return np.array_equal(max_vals_a, max_vals_b) and np.array_equal(max_cols_a, max_cols_b)

def validate_positive_integer(value: str, name: str) -> int:
    """Memvalidasi apakah input string adalah bilangan bulat positif."""
    try:
        val = int(value)
        if val <= 0:
            raise ValueError(f"{name} harus bilangan bulat positif > 0.")
        return val
    except ValueError as e:
        raise ValueError(f"Input tidak valid untuk {name}: {e}")

def format_result_summary(method: str, n: int, seed: int, workers: int,
                          mult_time: float, max_time: float, total_time: float,
                          checksum: int, is_valid: bool) -> str:
    """Memformat output ringkasan hasil untuk ditampilkan di terminal."""
    status = "VALID" if is_valid else "INVALID"
    
    summary = (
        f"Method                  : {method}\n"
        f"Matrix size             : {n} x {n}\n"
        f"Seed                    : {seed}\n"
        f"Workers                 : {workers}\n"
        f"Multiplication time     : {mult_time:.6f} seconds\n"
        f"Maximum search time     : {max_time:.6f} seconds\n"
        f"Total computation time  : {total_time:.6f} seconds\n"
        f"Checksum                : {checksum}\n"
        f"Validation status       : {status}"
    )
    return summary

def save_result_to_csv(filepath: str, method: str, n: int, workers: int, run_idx: int,
                       mult_time: float, max_time: float, total_time: float,
                       checksum: int, is_valid: bool) -> None:
    """Menyimpan hasil eksekusi ke dalam file CSV, menambahkan jika sudah ada."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    
    status = "VALID" if is_valid else "INVALID"
    
    with open(filepath, 'a') as f:
        if not file_exists:
            f.write("method,n,workers,run,multiplication_time,max_search_time,total_time,checksum,status\n")
        f.write(f"{method},{n},{workers},{run_idx},{mult_time:.6f},{max_time:.6f},{total_time:.6f},{checksum},{status}\n")
