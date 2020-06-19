import json
from ase.calculators.aims import Aims
from ibslib.calculators import Slurm,MBDBatchCalc, \
                               arjuna_arguments,mbd_settings

# mbd_settings["mbd_supercell_cutoff"] = 15
mbd_settings["mbd_scs_vacuum_axis"] = ".false. .false. .true."

# Directory with structure files to calculate
struct_dir = "SCF"
# Directory where calculations will take place
calc_dir = "batch_calc_MBD"
# Setup command for calculations
arjuna_arguments["command"] = "mpirun -np 7 /home/maromgroup/Software/MBD/DFT_MBD_AT_rsSCS.x geometry.xyz.xyz setting.in > mbd.out"
slurm = Slurm(arjuna_arguments)

calc = MBDBatchCalc(struct_dir, settings=mbd_settings, Slurm=slurm, calc_dir=calc_dir)
calc.calc(overwrite=True)
