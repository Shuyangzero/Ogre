# -*- coding: utf-8 -*-



# Default arguments for some computers. Can import these and use. 
# Make sure to specify a command.
arjuna_arguments = \
    {
        "-J": "ibslib",
        "-N": 1,
        "-n": 56,
        "--mem": 0,
        "-o": "j_%j.out",
        "-p": "cpu",
        "pre-command": "module load intel/18.0.3.222 genarris2",
        "command": "mpirun -np 56 /home/maromgroup/Software/bin/aims.180424.scalapack.mpi.x > aims.out",
    }
## Renaming for less typing while preserving previous API
arjuna = arjuna_arguments


hippolyta_arguments = \
    {
        "-J": "ibslib",
        "-N": 1,
        "-n": 32,
        "--mem": 0,
        "-o": "j_%j.out",
        "-p": "batch",
        "pre-command": None,
        "command": None,
    }
## Renaming for less typing while preserving previous API
hippolyta = hippolyta_arguments


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
## Renaming for less typing while preserving previous API
tin=tin_arguments