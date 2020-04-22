
__author__ = 'Manny Bier'

from .aims import *
from .slurm import *
from .mbd import *
from .k_grid_fn import *
from .slurm_arguments import *


### Some utilities for easy access below

def control_in(struct, relax=True, file_name="control.in",
               species_dir=""):
    """
    Easily generate control in file for input structure. 
    
    """
    if len(species_dir) == 0:
        if os.environ.get('AIMS_SPECIES_DIR') != None:
            species_dir = os.environ.get('AIMS_SPECIES_DIR')
        else:
            species_dir = input("Provide Aims Species Directory: ")
        
    if not relax:
        settings = tier1_SPE_settings
    else:
        settings = tier1_relaxed_settings
    
    settings["species_dir"] = species_dir
    a = Aims(**settings)
    
    atoms = struct.get_ase_atoms()
    
    a.write_control(atoms, file_name)
    a.write_species(atoms, file_name)
    
    
    

def make_submit(cluster="arjuna", file_name="Submit.sh",
                overwrite=False):
    """
    Function for quickly creating a correct Submit script, likely to be 
    modified by hand after creation.

    """
    args = eval(cluster)
    slurm = Slurm(args)
    slurm.write(file_name, overwrite=overwrite)
