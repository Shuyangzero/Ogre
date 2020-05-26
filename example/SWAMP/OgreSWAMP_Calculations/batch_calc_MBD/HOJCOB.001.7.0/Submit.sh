#!/bin/bash 
#SBATCH -J ibslib 
#SBATCH -N 1 
#SBATCH -n 12 
#SBATCH -o j_%j.out 
#SBATCH -p idle 
#SBATCH --mem-per-cpu=1000 

module load intel/18.0.3.222 genarris2
mpirun -np 7 /home/maromgroup/Software/MBD/DFT_MBD_AT_rsSCS.x geometry.xyz.xyz setting.in > mbd.out