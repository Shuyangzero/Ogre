module swap gnu intel/18.0.3.222
module swap openmpi impi/2018_Update_3
mpirun -np 56 /home/shuyangyang/bin/aims.180424.scalapack.mpi.x > aims.out
