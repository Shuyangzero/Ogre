

import json

import numpy as np

from ase.calculators.aims import Aims
from ibslib.io import read,write
from ibslib.calculators import Slurm,AimsBatchCalc, \
                               arjuna_arguments,tier1_SPE_settings, \
                               k_grid_24


#### Input directory ####
struct_dir="Raw"

#### Location of Calculations ####
calc_dir = "batch_calc"
tier1_SPE_settings["use_dipole_correction"] = True

#### Begin Calculation ####
slurm = Slurm(arjuna_arguments)
calc = AimsBatchCalc(struct_dir,
                     Slurm=slurm,
                     calc_dir=calc_dir,
                     k_grid_fn=k_grid_24,
                     aims_settings=tier1_SPE_settings)
calc.calc()


