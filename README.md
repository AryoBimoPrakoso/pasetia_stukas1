# Proyek Analisis Kinerja Paralel: Sistem Rekomendasi SnapGram

Proyek ini adalah implementasi **Studi Kasus 1: Sistem Rekomendasi "SnapGram" dan Analisis Kinerja Paralel** yang ditujukan untuk memenuhi tugas/UAS mata kuliah Algoritma dan Pengolahan Paralel.

## Deskripsi Studi Kasus SnapGram
Pada simulasi sistem SnapGram ini, kita merepresentasikan hubungan antar pengguna dan interaksi mereka dalam bentuk matriks. Operasi perkalian matriks (M × M) merepresentasikan penemuan jalur rekomendasi antara satu pengguna dengan pengguna lainnya. Pencarian nilai maksimum di setiap baris (beserta indeksnya) melambangkan rekomendasi teman terbaik atau konten paling relevan untuk masing-masing pengguna.

## Tujuan Proyek
1. Mengimplementasikan algoritma perkalian matriks secara berurutan (sekuensial).
2. Menerapkan pengolahan paralel secara *Shared-Memory*.
3. Menerapkan pengolahan paralel secara *Distributed-Memory*.
4. Menganalisis kinerja (Speedup dan Efisiensi) dari ketiga pendekatan tersebut dengan menyimpan hasil *benchmark* ke CSV dan memvisualisasikannya ke dalam bentuk grafik.

## Batasan Implementasi
- Seluruh kode ditulis dalam bahasa **Python 3**.
- Dilarang menggunakan fungsi perkalian instan seperti `np.dot` atau operator `@`. Perkalian matriks harus diimplementasikan manual menggunakan *loop* (`for`) agar algoritma paralel dapat diamati dan dikontrol.
- Metode Shared-Memory diimplementasikan menggunakan **Numba**.
- Metode Distributed-Memory diimplementasikan menggunakan **mpi4py** (Microsoft MPI).

## Penjelasan Pendekatan Pemrograman
- **Versi Sekuensial (`sequential.py`)**: Menggunakan `njit` dari pustaka Numba tanpa mengaktifkan mode paralelisasi. Numba mengompilasi kode Python (JIT) menjadi bahasa mesin agar tidak terlalu lambat dibandingkan Python murni.
- **Versi Shared-Memory (`parallel_numba.py`)**: Menggunakan Numba `@njit(parallel=True)` dan fungsi `prange` pada perulangan tingkat atas (baris). Pendekatan *work-sharing* `prange` ini secara konsep sangat mirip/ekuivalen dengan blok `#pragma omp parallel for` pada pustaka **OpenMP** (C/C++). Numba bukanlah OpenMP secara literal, tetapi abstraksinya sama untuk lingkungan Python.
- **Versi Distributed-Memory (`mpi_version.py`)**: Menggunakan pustaka `mpi4py` untuk saling berkirim pesan (Message Passing) antar *processes*. Setiap _process_ mempunyai ruang memori sendiri (terisolasi).

## Struktur Folder
```text
snapgram-python/
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── sequential.py
│   ├── parallel_numba.py
│   ├── mpi_version.py
│   ├── benchmark.py
│   └── create_graphs.py
├── tests/
│   └── test_algorithms.py
├── results/
│   ├── benchmark.csv (Dihasilkan oleh skrip)
│   ├── *.png         (Dihasilkan oleh skrip)
│   └── .gitkeep
├── scripts/
│   ├── run_sequential.bat
│   ├── run_numba_parallel.bat
│   ├── run_mpi.bat
│   └── run_all_benchmarks.bat
├── requirements.txt
├── README.md
└── .gitignore
```

## Penjelasan Algoritma
### Perkalian Matriks
Perkalian matriks diimplementasikan menggunakan tiga *nested-loop* (baris `i`, kolom `j`, elemen perantara `k`). Nilai `result[i, j]` ditambahkan terus menerus dengan hasil kali `matrix[i, k] * matrix[k, j]`. 

### Pencarian Maksimum per Baris
Setelah matriks hasil terbentuk, program akan melakukan iterasi pada setiap baris matriks. Nilai awal didefinisikan pada kolom indeks ke-0. Apabila pada iterasi kolom berikutnya (`j`) ditemukan nilai yang lebih besar, maka program akan menyimpan nilai dan indeks tersebut sebagai nilai maksimum baru untuk baris tersebut. Jika ada dua bilangan berukuran sama, bilangan yang pertama kali dijumpai yang akan dipakai.

### Pembagian Kerja (Shared-Memory Numba)
Pada `parallel_numba.py`, iterasi pada baris terluar `for i in prange(n):` akan didistribusikan secara dinamis ke seluruh _threads_ yang ada. Masing-masing thread memiliki memori bersama untuk mengakses `matrix`, namun setiap thread akan mengisi baris `i` yang berbeda pada `result`. 

### Pembagian Kerja (Distributed-Memory MPI)
Pada `mpi_version.py`, matriks sumber didistribusikan ke seluruh proses melalui perintah `Bcast`. Kemudian `Rank 0` menghitung proporsi baris untuk masing-masing proses berdasarkan *Size* / Jumlah Process. Proses menghitung perkalian hanya pada area baris yang menjadi tanggung jawabnya (`compute_local_rows`), kemudian Rank 0 menarik dan menggabungkan hasil perhitungannya menggunakan mekanisme `Gatherv`.

### Mekanisme Validasi (Triple-Check)
Sesuai dengan standar kebenaran ketat (*strict correctness*), program ini memvalidasi hasil eksekusi algoritma paralel (Numba dan MPI) dengan merujuk pada hasil dari versi Sequential Numba.
1. Script `sequential.py` akan menjadi **baseline**; ia otomatis menyimpan seluruh matriks referensi dan metadatanya (ukuran `n` dan `seed`) ke dalam file `results/reference_n{N}_seed{SEED}.npz`.
2. Skrip paralel akan secara serentak memuat file referensi tersebut.
3. Skrip paralel membandingkan tiga keluaran komputasi secara *element-wise* menggunakan fungsi `np.array_equal`: 
   - `result_matrix` == `ref_data['result_matrix']`
   - `max_values` == `ref_data['max_values']`
   - `max_indices` == `ref_data['max_indices']`
Jika ketiganya bernilai `True`, barulah iterasi pengujian dicap `VALID`.

## Persyaratan (Requirements)
* OS: Windows 10 atau Windows 11
* **Python 3.8+** terinstal di Windows.
* **Visual Studio Code** beserta ekstensi Python.
* **Microsoft MPI Runtime** (`msmpisetup.exe`) dan **Microsoft MPI SDK** (`msmpisdk.msi`).

## Persiapan Lingkungan (Environment)

### 1. Cara Membuat Virtual Environment
Buka terminal di root direktori proyek, kemudian ketik:
```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cara Instal Dependency
Setelah awalan `(.venv)` muncul, lakukan *upgrade* `pip` dan *install requirement*:
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cara Memeriksa MPI
Pastikan Microsoft MPI Runtime telah terinstal dengan menjalankan perintah ini (Anda mungkin perlu merestart terminal/VS Code setelah instalasi):
```bat
mpiexec -help
```
Jika memunculkan daftar panduan, instalasi MPI Anda berhasil.

## Cara Penggunaan

Gunakan _Test-Mode_ untuk menguji fungsi dengan ukuran matriks kecil `3x3` (seharusnya mengeluarkan status `VALID`).

### Mode Pengujian (Test-Mode)
```bat
python src/sequential.py --test-mode
python src/parallel_numba.py --test-mode --threads 4
mpiexec -n 2 python src/mpi_version.py --test-mode
```

### Menjalankan Versi Sequential
```bat
python src/sequential.py --size 512 --seed 12345
```

### Menjalankan Versi Numba Parallel
```bat
python src/parallel_numba.py --size 512 --threads 4 --seed 12345
```

### Menjalankan Versi MPI
```bat
mpiexec -n 4 python src/mpi_version.py --size 512 --seed 12345
```

### Menjalankan Benchmark Keseluruhan
Untuk otomatisasi, Anda dapat langsung menjalankan _batch script_ berikut yang akan mengotomatisasi pengujian berulang dan membuat file grafik.
```bat
.\scripts\run_all_benchmarks.bat
```
Atau manual melalui python:
```bat
python src/benchmark.py
```

### Membuat Grafik Secara Manual
```bat
python src/create_graphs.py
```

## Analisis Kinerja (Benchmark)

### Penjelasan Isi `benchmark.csv`
Skrip benchmark akan menyimpan datanya secara komulatif di `results/benchmark.csv` dengan kolom:
- `method`: Algoritma (sequential / numba_parallel / mpi)
- `n`: Ukuran matriks N
- `workers`: Jumlah Thread/Proses
- `run`: Pengulangan iterasi
- `multiplication_time`: Waktu (detik) memproses perkalian matriks
- `max_search_time`: Waktu (detik) memproses nilai maksimum baris
- `total_time`: Waktu keseluruhan (perkalian + search)
- `checksum`: Nilai validasi checksum matriks
- `status`: Indikator `VALID` / `INVALID`

### Rumus Perhitungan Kinerja
Grafik dan tabel dari `create_graphs.py` dihitung dengan rumus paralel berikut:
- **Speedup**: `Waktu Rata-rata Sekuensial / Waktu Rata-rata Paralel`
- **Efisiensi**: `(Speedup / Jumlah Worker) * 100%`

*(Semakin tinggi Speedup dan Efisiensi mendekati 100%, sistem paralel Anda semakin optimal).*

### Keterbatasan Program & Waktu Benchmark
Waktu komputasi yang direkam selama *benchmark* sifatnya relatif. Durasi perkalian sangat bergantung pada Spesifikasi CPU, Arsitektur, ketersediaan RAM, serta background proses lain di sistem Windows Anda. Waktu pada MPI kemungkinan akan terbebani oleh *overhead* komunikasi jaringan antar *node/process* (terutama pada *array* besar) yang seringkali menyebabkannya lebih lambat daripada *Shared-Memory* pada satu mesin komputer.

### Troubleshooting (Kemungkinan Error di Windows)
1. **`Cannot find module 'numba' / 'mpi4py'` di Editor VS Code** 
   - VS Code Anda belum memilih Interpreter virtual environment. Tekan `Ctrl+Shift+P` -> Pilih `Python: Select Interpreter` -> Cari direktori `./.venv/Scripts/python.exe`.
2. **`mpiexec is not recognized`**
   - Microsoft MPI belum dipasang atau PATH-nya belum terdaftar. Install dari situs resmi Microsoft, kemudian restart laptop atau terminal Anda.
3. **`ImportError: DLL load failed` saat import mpi4py**
   - Anda menggunakan Python versi lama yang rentan akan eror kompabilitas C-Extension Windows, gunakan **Python 3.10 atau 3.11**.
