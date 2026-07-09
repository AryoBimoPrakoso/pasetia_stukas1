"""
mpi_version.py
Implementasi algoritma perkalian matriks dan pencarian maksimum
menggunakan mpi4py untuk distributed-memory paralel (tanpa numba, Python murni).
"""
import argparse
import numpy as np
from mpi4py import MPI
from common import (
    generate_matrix, get_test_matrix, print_matrix_if_small, 
    calculate_checksum, format_result_summary, save_result_to_csv
)
import os
import pandas as pd


def compute_local_rows(matrix: np.ndarray, start_row: int, end_row: int) -> np.ndarray:
    """Melakukan perkalian matriks secara lokal hanya untuk baris tertentu (pure Python)."""
    n = matrix.shape[0]
    num_rows = end_row - start_row
    local_result = np.zeros((num_rows, n), dtype=np.int64)

    for i in range(num_rows):
        global_i = start_row + i
        for j in range(n):
            total = 0
            for k in range(n):
                total += int(matrix[global_i, k]) * int(matrix[k, j])
            local_result[i, j] = total
    return local_result


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


def find_local_max(local_result: np.ndarray):
    """Mencari nilai maksimum untuk baris yang menjadi tanggung jawab process ini (pure Python)."""
    num_rows = local_result.shape[0]
    n = local_result.shape[1]

    max_values = np.zeros(num_rows, dtype=np.int64)
    max_indices = np.zeros(num_rows, dtype=np.int32)

    for i in range(num_rows):
        current_max = local_result[i, 0]
        current_col = 0
        for j in range(1, n):
            if local_result[i, j] > current_max:
                current_max = local_result[i, j]
                current_col = j
        max_values[i] = current_max
        max_indices[i] = current_col

    return max_values, max_indices


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="MPI Matrix Multiplication and Max Search")
    parser.add_argument("--size", type=int, default=512, help="Ukuran matriks N")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--test-mode", action="store_true", help="Gunakan matriks 3x3 untuk pengujian")
    parser.add_argument("--run-idx", type=int, default=1, help="Indeks run untuk CSV")
    parser.add_argument("--csv", type=str, default="similarity_matrix_256.csv", help="Path File CSV matriks")
    parser.add_argument("--labels", type=str, default="users_256.csv", help="Nama file CSV label/identitas user")

    args = parser.parse_args()

    # Bungkus variabel `n` ke dalam array NumPy agar bisa di-Bcast dengan aman oleh mpi4py
    n_arr = np.array([args.size], dtype=np.int32)

    if args.test_mode:
        n_arr[0] = 3

    matrix = None

    # Rank 0 bertugas memuat atau membuat data asal
    if rank == 0:
        if args.test_mode:
            if size > 3:
                print("Test mode dibatasi maksimal 3 process untuk matriks 3x3.")
                comm.Abort(1)
            print("Menjalankan mode pengujian (Test Mode)...")
            matrix = get_test_matrix()
        else:
            csv_filename = args.csv
            print(f"Membaca data matriks dari file: {csv_filename}")

            if os.path.exists(csv_filename):
                # File berupa matriks angka murni: tidak ada header, tidak ada
                # kolom identitas (UserID/Persona).
                df = pd.read_csv(csv_filename, header=None)
                matrix = df.to_numpy(dtype=np.int64)
                n_arr[0] = matrix.shape[0]  # Update ukuran matrix berdasarkan real data CSV
                print(f"Berhasil memuat matriks dengan ukuran {n_arr[0]}x{n_arr[0]}")
            else:
                print(f"File {csv_filename} tidak ditemukan! Menggunakan data random fallback...")
                matrix = generate_matrix(n_arr[0], args.seed)
                # Pastikan tipe datanya int64 agar konsisten dengan proses local
                matrix = matrix.astype(np.int64)

    # --- KUNCI PERBAIKAN ---
    # Broadcast nilai `n` yang sebenarnya ke seluruh Rank agar sinkron
    comm.Bcast(n_arr, root=0)
    n = int(n_arr[0])

    if size > n:
        if rank == 0:
            print(f"Error: Jumlah process ({size}) lebih besar dari ukuran matriks ({n}).")
        MPI.Finalize()
        return

    # Rank selain 0 mengalokasikan ruang memori kosong untuk menerima data Broadcast matrix
    if rank != 0:
        matrix = np.empty((n, n), dtype=np.int64)

    # Sekarang aman menyiarkan matriks karena semua Rank memiliki dimensi `n` yang sama
    comm.Bcast(matrix, root=0)

    # Menghitung pembagian kerja (baris untuk tiap process)
    base_rows = n // size
    remainder = n % size

    counts = np.zeros(size, dtype=int)
    displs = np.zeros(size, dtype=int)

    offset = 0
    for i in range(size):
        counts[i] = base_rows + (1 if i < remainder else 0)
        displs[i] = offset
        offset += counts[i]

    local_count = counts[rank]
    local_start = displs[rank]
    local_end = local_start + local_count

    # Sinkronisasi sebelum menghitung waktu
    comm.Barrier()

    # Mulai menghitung waktu (lokal)
    t0_mult = MPI.Wtime()
    local_result = compute_local_rows(matrix, local_start, local_end)
    t1_mult = MPI.Wtime()
    local_mult_time = t1_mult - t0_mult

    t0_max = MPI.Wtime()
    local_max_values, local_max_indices = find_local_max(local_result)
    t1_max = MPI.Wtime()
    local_max_time = t1_max - t0_max

    local_total_time = local_mult_time + local_max_time

    # Mengumpulkan waktu terlama dari seluruh process (reduksi MAX)
    global_mult_time = comm.reduce(local_mult_time, op=MPI.MAX, root=0)
    global_max_time = comm.reduce(local_max_time, op=MPI.MAX, root=0)
    global_total_time = comm.reduce(local_total_time, op=MPI.MAX, root=0)

    # Mengumpulkan (gather) matriks hasil ke Rank 0
    if rank == 0:
        global_result = np.zeros((n, n), dtype=np.int64)
        global_max_values = np.zeros(n, dtype=np.int64)
        global_max_indices = np.zeros(n, dtype=np.int32)
    else:
        global_result = None
        global_max_values = None
        global_max_indices = None

    # Menyiapkan perhitungan elemen untuk Gatherv matriks (baris * kolom)
    recv_counts_matrix = counts * n
    recv_displs_matrix = displs * n

    comm.Gatherv(local_result, [global_result, recv_counts_matrix, recv_displs_matrix, MPI.INT64_T], root=0)
    comm.Gatherv(local_max_values, [global_max_values, counts, displs, MPI.INT64_T], root=0)
    comm.Gatherv(local_max_indices, [global_max_indices, counts, displs, MPI.INT32_T], root=0)

    # Rank 0 memvalidasi dan mencetak hasil
    if rank == 0:
        checksum = calculate_checksum(global_result)

        is_valid = True
        if args.test_mode:
            expected_result = np.array([
                [30, 36, 42],
                [66, 81, 96],
                [102, 126, 150]
            ], dtype=np.int64)
            expected_max_values = np.array([42, 96, 150], dtype=np.int64)
            expected_max_indices = np.array([2, 2, 2], dtype=np.int32)

            is_valid = (np.array_equal(global_result, expected_result) and 
                        np.array_equal(global_max_values, expected_max_values) and 
                        np.array_equal(global_max_indices, expected_max_indices))
        else:
            ref_file = f"results/reference_n{n}_seed{args.seed}.npz"
            if os.path.exists(ref_file):
                ref_data = np.load(ref_file)
                is_valid = (np.array_equal(global_result, ref_data['result_matrix']) and
                            np.array_equal(global_max_values, ref_data['max_values']) and
                            np.array_equal(global_max_indices, ref_data['max_indices']))
            else:
                print("Peringatan: File referensi sekuensial tidak ditemukan.")

        print(format_result_summary(
            method="MPI",
            n=n,
            seed=args.seed,
            workers=size,
            mult_time=global_mult_time,
            max_time=global_max_time,
            total_time=global_total_time,
            checksum=checksum,
            is_valid=is_valid
        ))

        labels_df = None if args.test_mode else load_labels(args.labels)

        if labels_df is not None:
            narrative_values, narrative_indices = find_row_max_excluding_self(global_result)

            print("\nLima contoh hasil kemiripan (naratif, tanpa self-match):")
            limit = min(5, n)
            for narrative in build_similarity_narrative(labels_df, narrative_indices[:limit], narrative_values[:limit]):
                print(narrative)

            # Simpan laporan naratif lengkap untuk semua user
            all_narratives = build_similarity_narrative(labels_df, narrative_indices, narrative_values)
            os.makedirs('results', exist_ok=True)
            report_path = "results/similarity_narrative_mpi.csv"
            pd.DataFrame({"Ringkasan_Kemiripan": all_narratives}).to_csv(report_path, index=False)
            print(f"\nLaporan naratif lengkap disimpan di: {report_path}")
        else:
            print("\nLima hasil baris pertama (maksimum dan indeks kolom):")
            limit = min(5, n)
            for i in range(limit):
                print(f"Row {i}: max = {global_max_values[i]}, column = {global_max_indices[i]}")

        if not args.test_mode:
            save_result_to_csv(
                filepath="results/benchmark.csv",
                method="mpi",
                n=n,
                workers=size,
                run_idx=args.run_idx,
                mult_time=global_mult_time,
                max_time=global_max_time,
                total_time=global_total_time,
                checksum=checksum,
                is_valid=is_valid
            )

    MPI.Finalize()

if __name__ == "__main__":
    main()