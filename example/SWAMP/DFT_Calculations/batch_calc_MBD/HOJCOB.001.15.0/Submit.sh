#!/bin/bash
#SBATCH -J mbd
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o j_%j.out
#SBATCH -p idle
#SBATCH --mem=0

module load intel/18.0.3.222 genarris2
mpirun -np 2 /home/maromgroup/Software/MBD/DFT_MBD_AT_rsSCS.x geometry.xyz.xyz setting.in > mbd.out
