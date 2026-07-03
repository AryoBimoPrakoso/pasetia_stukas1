@echo off
mpiexec -n 4 python src\mpi_version.py --size 512 --seed 12345
