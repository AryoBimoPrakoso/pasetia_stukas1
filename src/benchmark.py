"""
benchmark.py
Skrip otomatisasi untuk menjalankan skenario pengujian 
dan mengumpulkan hasil benchmark ke dalam CSV.
"""
import subprocess
import os
import sys

def run_command(cmd, description):
    print(f"Menjalankan: {description}")
    print(f"Command: {cmd}")
    try:
        # Gunakan shell=True di Windows agar mempermudah pemanggilan command
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: Pengujian {description} gagal dengan exit code {e.returncode}.")
    print("-" * 50)

def main():
    print("=== Memulai Otomatisasi Benchmark ===")
    
    # Inisialisasi ulang CSV di awal benchmark
    csv_file = "results/benchmark.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, 'w') as f:
        f.write("method,n,workers,run,multiplication_time,max_search_time,total_time,checksum,status\n")
    print(f"File {csv_file} berhasil di-reset.")
    
    # Konfigurasi pengujian
    sizes = [128, 256, 512, 1024]
    runs = 3
    seed = 12345
    
    # Menentukan Python mana yang dipakai (utamakan .venv)
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    for n in sizes:
        print(f"\n>>> BENCHMARK UKURAN MATRIKS: {n}x{n} <<<")
        for run_idx in range(1, runs + 1):
            # 1. Sequential (1 worker)
            cmd_seq = f"{python_exe} src/sequential.py --size {n} --seed {seed} --run-idx {run_idx}"
            run_command(cmd_seq, f"Sequential Numba (Size {n}, Run {run_idx})")
            
            # 2. Numba Parallel (2, 4, 8 threads)
            for threads in [2, 4, 8]:
                cmd_numba = f"{python_exe} src/parallel_numba.py --size {n} --threads {threads} --seed {seed} --run-idx {run_idx}"
                run_command(cmd_numba, f"Numba Parallel {threads} Threads (Size {n}, Run {run_idx})")
                
            # 3. MPI Version (2, 4 process)
            for procs in [2, 4]:
                cmd_mpi = f"mpiexec -n {procs} {python_exe} src/mpi_version.py --size {n} --seed {seed} --run-idx {run_idx}"
                run_command(cmd_mpi, f"MPI {procs} Processes (Size {n}, Run {run_idx})")
                
    print("\n=== Seluruh Benchmark Selesai ===")
    print("Hasil tersimpan di: results/benchmark.csv")

if __name__ == "__main__":
    main()
