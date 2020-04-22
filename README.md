# Ogre 
<img src="imgs/logo.png" alt="logo" align="bottom">

Ogre is written in Python 3 and uses  the  ase, and  pymatgen libraries.   The  code  is under a BSD-3license license.  Ogre takes a bulk crystalstructure as input. Several common structure input formats are supported, including CIF, VASP POSCAR files, and FHI-aims geometry.in files. Ogre generates surface slabs by cleaving thebulk crystal along a user-specified Miller plane. Ogre outputs surface slab models with the number of layers and amount of vacuum requested by the user. 
## Installation
```bash
conda create -n ogre python=3.6
conda activate ogre
python setup.py install
```
This code might crash if you use the newest version of pymatgen, so please use the version specified in setup.py

## Example
All configurations are stored in the .config file
Ogre is for cleaving the surfaces, ogreSWAMP is for plotting surface energy convergence plots and Wulff diagram.


ogre.config:


```bash
[io]
structure_path = ./structures/relaxed_structures/aspirin.cif
structure_name = aspirin
format = FHI
; Format can be FHI, VASP or CIF
[methods]
cleave_option = 1
; 0: cleave for a single surface, 1: cleave for surface energy calculations
[parameters]
layers = 1-9 13 14
; Layers could be specified as combination of start-end or separate numbers by space
vacuum_size = 40
highest_index = 1
; Only meaningful when cleave_option = 1
supercell_size = None
miller_index = 1 0 0
; Only meaningful when cleave_option = 0
```


ogreSWAMP.config:
```
[io]
scf_path = example/SCF
structure_path = structures/relaxed_structures/ASPIRIN.cif
structure_name = aspirin
[methods]
fitting_method = 0
; 0: linear method, 1: Boettger method.
[Wulff]
Wulff_plot = True
; Wehther to plot the Wulff diagram
projected_direction = 1 1 1
; projected_direction for Wulff diagram
[convergence]
threshhold = 0.35
consecutive_step = 2
; If for N consecutive steps, the difference is smaller than threshhold, then it is converged.
```

To run ogre
```bash
python runOgre.py --filename ogre.config
```
To launch DFT calculations
```bash
#Launch PBE+TS, please modify the setting in the script, same for the following scripts.
python AimsBatchCAlc.py

#Extract the data from PBE+TS and store in .json file
python AimsExtractor.py

#Launch PBE+MBD
python MBDBatchCalc.py

#Store data in .json file
python MBDExtract.py
```
To run ogreSWAMP
```bash
python runOgreSWAMP.py --filename ogreSWAMP.config
```
