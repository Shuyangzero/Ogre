#!/bin/bash 
#SBATCH -J HOJCOB 
#SBATCH -N 16 
#SBATCH -n 512 
#SBATCH -o j_%j.out 
#SBATCH -t 8:00:00 
#SBATCH -q regular 
#SBATCH -A m3578 
#SBATCH --constraint=knl 

export OMP_NUM_THREADS=1

srun -n 512 /global/cfs/cdirs/m3578/aims/aims.180424.scalapack.mpi.x > aims.out