"""
sequential.py
Implementasi algoritma perkalian matriks dan pencarian maksimum
secara sekuensial menggunakan Python murni (tanpa numba, tanpa JIT).
"""
import argparse
import time
import numpy as np
from common import (
    generate_matrix, get_test_matrix, print_matrix_if_small, 
    calculate_checksum, format_result_summary, save_result_to_csv
)
import os
import pandas as pd  # <--- Tambahkan ini di bagian atas bersama import lainnya


def matrix_multiply_sequential(matrix: np.ndarray) -> np.ndarray:
    """Melakukan perkalian matriks secara sekuensial dengan 3 nested loop (pure Python)."""
    n = matrix.shape[0]
    result = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            total = 0
            for k in range(n):
                total += int(matrix[i, k]) * int(matrix[k, j])
            result[i, j] = total
    return result


def find_row_max_excluding_self(result_matrix: np.ndarray):
    """Mencari nilai & indeks kolom maksimum per baris, mengecualikan elemen
    diagonal (self-similarity) agar hasil narasi lebih bermakna."""
    n = result_matrix.shape[0]
    max_values = np.zeros(n, dtype=np.int64)
    max_indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        current_max = None
        current_col = -1
        for j in range(n):
            if j == i:
                continue
            val = result_matrix[i, j]
            if current_max is None or val > current_max:
                current_max = val
                current_col = j
        max_values[i] = current_max
        max_indices[i] = current_col
    return max_values, max_indices


def load_labels(labels_path: str):
    """Memuat file label (UserID, Umur, Gender, Daerah, Kategori) jika tersedia."""
    if labels_path and os.path.exists(labels_path):
        return pd.read_csv(labels_path)
    return None


def describe_user(labels_df, idx: int) -> str:
    """Membuat deskripsi singkat untuk satu user berdasarkan indeks baris."""
    if labels_df is None or idx >= len(labels_df):
        return f"User#{idx}"
    row = labels_df.iloc[idx]
    return f"{row['UserID']} ({row['Kategori']}, {row['Daerah']})"


def build_similarity_narrative(labels_df, max_indices, max_values=None) -> list:
    """Mengubah hasil (index kolom termirip per baris) menjadi kalimat naratif.
    Jika max_values diberikan, ditambahkan catatan skor mentah di akhir kalimat."""
    narratives = []
    for i, j in enumerate(max_indices):
        src = describe_user(labels_df, i)
        tgt = describe_user(labels_df, int(j))
        if max_values is not None:
            narratives.append(f"{src} lebih mirip ke {tgt} (catatan: nilai kemiripan = {max_values[i]})")
        else:
            narratives.append(f"{src} lebih mirip ke {tgt}")
    return narratives


def find_row_max_sequential(result_matrix: np.ndarray):
    """Mencari nilai maksimum dan indeks kolomnya untuk setiap baris (pure Python)."""
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


def main():
    parser = argparse.ArgumentParser(description="Sequential Matrix Multiplication and Max Search")
    parser.add_argument("--size", type=int, default=512, help="Ukuran matriks N")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--test-mode", action="store_true", help="Gunakan matriks 3x3 untuk pengujian")
    parser.add_argument("--run-idx", type=int, default=1, help="Indeks run untuk iterasi benchmark")
    parser.add_argument("--csv", type=str, default="similarity_matrix_256.csv", help="Nama file CSV matriks input")
    parser.add_argument("--labels", type=str, default="users_256.csv", help="Nama file CSV label/identitas user")

    args = parser.parse_args()

    # Validasi input
    if args.size <= 0:
        print("Error: Ukuran matriks harus lebih besar dari 0.")
        return

    if args.test_mode:
        print("Menjalankan mode pengujian (Test Mode)...")
        matrix = get_test_matrix()
        args.size = 3
    else:
        # --- Baca matriks similarity dari CSV ---
        csv_filename = args.csv
        print(f"Membaca data matriks dari file: {csv_filename}")

        if os.path.exists(csv_filename):
            # File berupa matriks angka murni: tidak ada header, tidak ada
            # kolom identitas (UserID/Persona) seperti dataset SnapGram sebelumnya.
            df = pd.read_csv(csv_filename, header=None)
            matrix = df.to_numpy(dtype=np.int64)

            args.size = matrix.shape[0]  # otomatis mengikuti jumlah baris CSV
            print(f"Berhasil memuat matriks dengan ukuran {args.size}x{args.size}")
        else:
            print(f"File {csv_filename} tidak ditemukan! Menggunakan data random fallback...")
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
        method="Sequential Python",
        n=args.size,
        seed=args.seed,
        workers=1,
        mult_time=mult_time,
        max_time=max_time,
        total_time=total_time,
        checksum=checksum,
        is_valid=is_valid
    ))

    labels_df = None if args.test_mode else load_labels(args.labels)

    if labels_df is not None:
        narrative_values, narrative_indices = find_row_max_excluding_self(result_matrix)

        print("\nLima contoh hasil kemiripan (naratif, tanpa self-match):")
        limit = min(5, args.size)
        for narrative in build_similarity_narrative(labels_df, narrative_indices[:limit], narrative_values[:limit]):
            print(narrative)

        # Simpan laporan naratif lengkap untuk semua user
        all_narratives = build_similarity_narrative(labels_df, narrative_indices, narrative_values)
        os.makedirs('results', exist_ok=True)
        report_path = "results/similarity_narrative_sequential.csv"
        pd.DataFrame({"Ringkasan_Kemiripan": all_narratives}).to_csv(report_path, index=False)
        print(f"\nLaporan naratif lengkap disimpan di: {report_path}")
    else:
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