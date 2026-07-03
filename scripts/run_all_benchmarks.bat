@echo off
echo Memeriksa ketersediaan Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python tidak ditemukan. Pastikan sudah di-install dan ada di PATH.
    exit /b 1
)

echo Memeriksa ketersediaan Microsoft MPI...
mpiexec -help >nul 2>&1
if %errorlevel% neq 0 (
    echo mpiexec tidak ditemukan! Silakan install Microsoft MPI Runtime ^& SDK terlebih dahulu.
    echo Buka: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
    exit /b 1
)

echo Menjalankan benchmark...
python src\benchmark.py

echo.
echo Membuat grafik visualisasi...
python src\create_graphs.py

echo Selesai!
pause
