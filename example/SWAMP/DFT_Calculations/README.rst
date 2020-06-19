Introduction
------------
This document will teach you how to run a large number of FHI-aims calculations using ``ibslib``. 
The way we run the script will require Slurm to be installed on the computer cluster and 
may require some settings contained within the script to be ameanable to the Slurm scheduler. 

The general workflow for FHI-aims calculations for Ogre is as follows: 1. Submit FHI-aims calculations using ``AimsBatchCalc.py``. Extract results using ``AimsExtractor.py`` in a folder of ``json`` files. If desired, calculation the MBD energy using ``MBDBatchCalc.py`` and extract the results into the same ``json`` files using ``MBDExtract.py``. 

An example of the outputs obtained from these operatios has been placed in this directory for the ``001`` surface of the bulk HMX crystal. The results of the FHI-aims SCF calculations can be found in ``batch_calc_scf``. Results of the extraction from these calculations are then stored in ``SCF``. The MBD calculations took place in ``batch_calc_MBD`` and the results are extracted back into the ``SCF`` folder. 


AimsBatchCalc
-------------
This section will go over the use of ``AimsBatchCalc.py`` in the Scripts folder.
This script ties together three ``Calculators``, the ``Slurm`` calculator and 
the ``AimsBatchCalc`` calculator from ``ibslib``, and the ``Aims`` calculator 
from ``ASE``. We will go into detail about the responsibilities of each 
calculator.

Slurm
^^^^^
The ``Slurm`` calculator is responsible for creating the **submission** script, 
and calling ``sbatch``, which submits the calculation to the queue. The ``Slurm``
class requires a single argument that provides the settings used to create the 
submission script. 

``slurm_arguments``
    Dictionary with keys,value pairs that correspond to the desired Slurm options. In addition, there are two special keys. ``command`` is printed as the last line in the submission script and defines the job command. ``pre-command`` is used for any environment modifiers to be printed directly before the ``command``.

Within ``ibslib``, there are default ``slurm_arguments`` stored in ``ibslib.calculators. 
Below are two examples.::

    arjuna_arguments = \
        {
            "-J": "ibslib",
            "-N": 1,
            "-n": 56,
            "--mem": 0,
            "-o": "j_%j.out",
            "-p": "cpu",
            "pre-command": None,
            "command": None,
        }
        
    tin_arguments = \
        {
            "-J": "ibslib",
            "-N": 1,
            "-n": 12,
            "--mem": 0,
            "-o": "j_%j.out",
            "-p": "Manny_is_supercool",
            "pre-command": None,
            "command": None,
        }
    

As you can see, the arguments are stored as a dictionary where the key is the 
Slurm option and value is what should be printed after. Here is an example, 
of using the ``arjuna_arguments``::

    >>> from ibslib.calculators import Slurm,arjuna_arguments
    >>> arjuna_arguments["command"] = 'echo "Hello World"'
    >>> slurm = Slurm(arjuna_arguments)
    >>> slurm.write("Submit.sh")

This is the ``Submit.sh`` file created by the above commands::

    #!/bin/bash
    #SBATCH -J ibslib
    #SBATCH -N 1
    #SBATCH -n 56
    #SBATCH --mem=0
    #SBATCH -o j_%j.out
    #SBATCH -p cpu
    
    echo "Hello World"
    
**NOTE**: A ``command`` must be supplied in the arguments, or else instantiation
of the ``Slurm`` class will have an error that says 
``A valid command was not specified in arguments to the slurm class.``


Aims
^^^^
You can learn more about the ``ASE`` calculator for FHI-aims by going to the
``ASE`` documentation_. In our case, we will only be using this calculator 
to create the ``control.in`` and ``geometry.in`` for the calculation. 
Initialization of the calculator requires specifying all of the DFT settings
that you would like to use. Then, the species information in the control file 
is added automatically. Our standard relaxation settings can be imported from 
``ibslib`` using ``from ibslib.calculators import tier1_relaxed_settings``.

.. _documentation: https://wiki.fysik.dtu.dk/ase/ase/calculators/FHI-aims.html

AimsBatchCalc
^^^^^^^^^^^^^
``ibslib.calculators.AimsBatchCalc`` ties together the ``Slurm`` calculator 
and the ``ASE`` ``Aims`` calculator. You may instantiate the calculator as 
follows.::

    >>> from ibslib.calculators import AimsBatchCalc,Slurm,tier1_relaxed_settings,arjuna_arguments
    >>> from ase.calculators.aims import Aims
    >>> arjuna_arguments["command"] = 'mpirun -np 56 /home/maromgroup/Software/bin/aims.180424.scalapack.mpi.x > aims.out'
    >>> struct_dir = "directory_of_structures_to_relax"
    >>> calc_dir = "directory_where_calculations_will_take_place"
    >>> AimsBatchCalc(struct_dir, slurm=Slurm(arjuna_arguments), calc_dir=calc_dir, aims_settings=tier1_relaxed_settings)

Note that the ``arjuna_argument["command"]`` in this case is used to run an FHI-aims 
calculation on Arjuna. The definitions of the arguments for ``AimsBatchCalc`` are 
as follows:

``struct_dir``
    Path to the directory of structures to be calculated.

``aims_settings``
    Arguments passed to ``ase.calculators.aims.Aims`` because ``AimsBatchCalc`` is an inherited class of this ``ase`` module. 

``Slurm``
    ``ibslib.calculators.slurm.Slurm`` class initialized with all parameters desired for the submission file. 
    
``calc_dir``
    Path to the directory where calculations should take place. Default is to use a folder called ``batch_calc``.


Additionally, it may be necessary to modify some of the tier1_relaxed_settings for your computer environment. In particular, change the value of the ``tier1_relaxed_settings["species_dir"]`` to the location of the light or tight species directory for your FHI-aims installation. 


Extracting Results
------------------
Extracting results consists of taking the results from an FHI-aims calculation directory, parsing the relevant information, and turning this into a ``Structure.json`` file. There's a speficic module in ``ibslib.io`` called ``AimsExtractor`` just for this purpose. Upon completion, the ``AimsExtractor.run_extraction()`` method will return a ``StructDict`` of all the ``Structures``. The user may write this to new folder in any file format that they wish. Below are a description of the arguments for the ``AimsExctractor``::

    AimsExtractor(calc_dir, aims_property=['energy', 'time', "sg"], 
    energy_name='energy_tier1_relaxed', log_file=None, 
    name_func=name_from_path, symprec=1.0)
    
The definition of each argument is as follows:

``calc_dir``
    Directory of FHI-aims calculations that you wish to create ``Structure.json`` files for.

``aims_property``
    Properties you hope to extract from the FHI-aims calcuation. Acceptable strings for this are ``energy``, ``vdw_energy``, ``time``, ``sg`` or ``space_group``, ``hirshfeld_volumes`` or ``atom_volumes``. 

``energy_name``
    The energy value will be saved with this name in the ``Structure.json`` property section. Typically, this energy name should be descriptive of the settings that were used for the FHI-aims calculation.
 
``log_file``
    If ``None``, the extractor will use ``STDOUT`` to print information from the extractor operation. Otherwise, if a name is specified, then the extractor will write its information to that file. 
 
``name_func``
    Allows the user to create arbitrary ways to define the ``Structure.struct_id``. The default is a function called ``name_from_path`` that uses the last string of each FHI-aims calculation directory path. However, the user may choose to specify any type of naming scheme that may or may not be dependent on the FHI-aims calculation path.
    
``symprec``
    Symmetry precision for space group identification using Pymatgen symmetry analyzer, which is just a wrapper around ``spglib``. 

    
