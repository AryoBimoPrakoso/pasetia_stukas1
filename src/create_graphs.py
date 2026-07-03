"""
create_graphs.py
Membaca file benchmark.csv menggunakan Pandas dan
membuat visualisasi grafik perbandingan kinerja.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    csv_file = "results/benchmark.csv"
    if not os.path.exists(csv_file):
        print(f"File {csv_file} tidak ditemukan. Jalankan benchmark terlebih dahulu.")
        return
        
    print("Membaca data benchmark...")
    df = pd.read_csv(csv_file)
    
    # 1. Validasi jumlah baris total harus 72
    if len(df) != 72:
        print(f"ERROR: Total baris data tidak sama dengan 72 (ditemukan {len(df)} baris).")
        return
        
    # 2. Validasi duplikat
    duplicates = df[df.duplicated(subset=['method', 'n', 'workers', 'run'], keep=False)]
    if not duplicates.empty:
        print("ERROR: Ditemukan data duplikat berdasarkan method, n, workers, dan run:")
        print(duplicates)
        return
        
    # 3. Validasi run {1, 2, 3}
    run_counts = df.groupby(['method', 'n', 'workers'])['run'].nunique()
    if not all(run_counts == 3):
        print("ERROR: Ada konfigurasi yang tidak memiliki tepat 3 run unik.")
        return
        
    run_values = set(df['run'].unique())
    if run_values != {1, 2, 3}:
        print(f"ERROR: Nomor run harus 1, 2, dan 3. Ditemukan: {run_values}")
        return
        
    # Filter data yang VALID saja
    df = df[df['status'] == 'VALID']
    if df.empty:
        print("Tidak ada data valid untuk dibuatkan grafik.")
        return
        
    # Hitung rata-rata waktu berdasarkan method, n, dan workers
    agg_df = df.groupby(['method', 'n', 'workers']).agg({
        'total_time': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Rapikan nama kolom
    agg_df.columns = ['method', 'n', 'workers', 'time_mean', 'time_min', 'time_max', 'time_std']
    
    # Pisahkan data berdasar metode
    seq_data = agg_df[agg_df['method'] == 'sequential']
    
    # Buat direktori output
    os.makedirs("results", exist_ok=True)
    
    # 1. Grafik Waktu Eksekusi terhadap Ukuran Matriks
    plt.figure(figsize=(10, 6))
    for method in ['sequential', 'numba_parallel', 'mpi']:
        method_data = agg_df[agg_df['method'] == method]
        if not method_data.empty:
            # Ambil waktu tercepat (worker terbanyak/paling optimal) untuk tiap n
            best_times = method_data.loc[method_data.groupby('n')['time_mean'].idxmin()]
            best_times = best_times.sort_values('n')
            plt.plot(best_times['n'], best_times['time_mean'], marker='o', label=method)
            
    plt.title("Average Execution Time vs Matrix Size (Best Configuration)")
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Total Time (Seconds)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/execution_time.png", bbox_inches='tight')
    plt.close()
    
    # Persiapkan data Speedup dan Efisiensi
    # Untuk menghitung speedup, butuh base time dari sequential untuk tiap n
    base_times = seq_data.set_index('n')['time_mean'].to_dict()
    
    # 2. Grafik Speedup terhadap Jumlah Worker (Fokus matriks terbesar saja)
    largest_n = agg_df['n'].max()
    plt.figure(figsize=(10, 6))
    
    for method in ['numba_parallel', 'mpi']:
        subset = agg_df[(agg_df['method'] == method) & (agg_df['n'] == largest_n)].sort_values('workers')
        if not subset.empty and largest_n in base_times:
            speedup = base_times[largest_n] / subset['time_mean']
            plt.plot(subset['workers'], speedup, marker='s', label=method)
            
    # Garis ideal
    max_workers_plot = agg_df[agg_df['n'] == largest_n]['workers'].max()
    if pd.notna(max_workers_plot) and max_workers_plot > 1:
        plt.plot([1, max_workers_plot], [1, max_workers_plot], 'k--', label='Ideal Speedup')
        
    plt.title(f"Speedup vs Number of Workers (Matrix Size: {largest_n}x{largest_n})")
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/speedup.png", bbox_inches='tight')
    plt.close()
    
    # 3. Grafik Efisiensi terhadap Jumlah Worker
    plt.figure(figsize=(10, 6))
    for method in ['numba_parallel', 'mpi']:
        subset = agg_df[(agg_df['method'] == method) & (agg_df['n'] == largest_n)].sort_values('workers')
        if not subset.empty and largest_n in base_times:
            speedup = base_times[largest_n] / subset['time_mean']
            efficiency = (speedup / subset['workers']) * 100
            plt.plot(subset['workers'], efficiency, marker='^', label=method)
            
    plt.axhline(y=100, color='k', linestyle='--', label='Ideal Efficiency')
    plt.title(f"Efficiency vs Number of Workers (Matrix Size: {largest_n}x{largest_n})")
    plt.xlabel("Number of Workers")
    plt.ylabel("Efficiency (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/efficiency.png", bbox_inches='tight')
    plt.close()
    
    # 4. Grafik Perbandingan Sequential, Numba Parallel, MPI
    plt.figure(figsize=(10, 6))
    compare_data = agg_df[agg_df['n'] == largest_n]
    if not compare_data.empty:
        idx = compare_data.groupby('method')['time_mean'].idxmin()
        best_compare = compare_data.loc[idx]
        
        plt.bar(best_compare['method'], best_compare['time_mean'])
        plt.title(f"Method Comparison - Average Total Time (Matrix Size {largest_n})")
        plt.xlabel("Method")
        plt.ylabel("Time (Seconds)")
        plt.grid(axis='y')
        plt.savefig("results/method_comparison.png", bbox_inches='tight')
        plt.close()
        
    print("Pembuatan grafik selesai. Hasil disimpan di folder results/")
    
    # Menampilkan tabel ringkasan
    print("\nRingkasan Analisis Kinerja:")
    print("=" * 100)
    agg_df['speedup'] = agg_df.apply(
        lambda row: base_times.get(row['n'], np.nan) / row['time_mean'] 
        if row['method'] != 'sequential' else 1.0, axis=1)
    agg_df['efficiency'] = (agg_df['speedup'] / agg_df['workers']) * 100
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(agg_df.to_string(index=False))

if __name__ == "__main__":
    main()
