# Ogre 
<img src="imgs/logo.png" alt="logo" align="bottom">

Ogre is written in Python and interfaces with the FHI-aims code to calculate surface energies at the level of density functional theory (DFT). The input of Ogre is the geometry of the bulk molecular crystal. The surface is cleaved from the bulk structure with the molecules on the surface kept intact. A slab model is constructed according to the user specifications for the number of molecular layers and the length of the vacuum region. Ogre automatically identifies all symmetrically unique surfaces for the user-specified Miller indices and detects all possible surface terminations. Ogre includes utilities, OgreSWAMP, to analyze the surface energy convergence and Wulff shape of the molecular crystal. 

## Citation
Please cite
```
Yang, Shuyang, et al. "Ogre: A Python package for molecular crystal surface generation with applications to surface energy and crystal habit prediction." The Journal of Chemical Physics 152.24 (2020): 244122.
```
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
structure_path = ./structures/relaxed_structures/TETCEN.cif
structure_name = TETCEN_test
format = FHI
; Format can be FHI, VASP or CIF
[methods]
cleave_option = 1
;0: cleave a single surface, 1: cleave surfaces for surface energy calculations
[parameters]
layers = 1-6
; Layers could be specified as combination of start-end or separate numbers by space
vacuum_size = 40
highest_index = 3
; Only needed in cleave_option = 1， ignored in cleave_option = 0
supercell_size = None
miller_index = 3 3 3
; Only needed in cleave_option = 0, ignored in cleave_option = 1
desired_num_of_molecules_oneLayer = 0
; Set to 0 if you don't want to change one layer structure. Default is 0 (highly recommended)
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
