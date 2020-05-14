# Ogre 
<img src="imgs/logo.png" alt="logo" align="bottom">

Ogre is written in Python and interfaces with the FHI-aims code to calculate surface energies at the level of density functional theory (DFT). The input of Ogre is the geometry of the bulk molecular crystal. The surface is cleaved from the bulk structure with the molecules on the surface kept intact. A slab model is constructed according to the user specifications for the number of molecular layers and the length of the vacuum region. Ogre automatically identifies all symmetrically unique surfaces for the user-specified Miller indices and detects all possible surface terminations. Ogre includes utilities, OgreSWAMP, to analyze the surface energy convergence and Wulff shape of the molecular crystal. 

## Installation
```bash
conda create -n ogre python=3.6
conda activate ogre
python setup.py install
```
This code might crash if you use the newest version of pymatgen, so please use the version specified in setup.py

## Example
All configurations are stored in the .config file.


Ogre is for cleaving the surfaces, ogreSWAMP is for plotting surface energy convergence plots and Wulff diagram.


ogre.config:


```
[io]
structure_path = ./structures/relaxed_structures/aspirin.cif
structure_name = aspirin
format = FHI
; Format can be FHI, VASP or CIF
[methods]
cleave_option = 1
; 0: cleave a single surface, 1: cleave surfaces for urface energy calculations
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
To launch DFT calculations, use scripts under scripts/
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
