# Ogre 
<img src="imgs/logo.png" alt="logo" align="bottom">

Ogre is written in Python 3 and uses  the  ase, and  pymatgen libraries.   The  code  is under a BSD-3license license.  Ogre takes a bulk crystalstructure as input. Several common structure input formats are supported, including CIF, VASP POSCAR files, and FHI-aims geometry.in files. Ogre generates surface slabs by cleaving thebulk crystal along a user-specified Miller plane. Ogre outputs surface slab models with the number of layers and amount of vacuum requested by the user. 
## Installation
```bash
conda create -n ogre python=3.7
conda activate ogre
python setup.py install
```
This code might crash if you use the newest version of pymatgen, so please use the version specified in setup.py

## Example

For import paths to work correctly, do
```bash
source init.sh
```
All configurations are stored in the .config file

ogre.config:

```bash
[io]
structure_path = ./structures/relaxed_structures/aspirin.cif
structure_name = aspirin
format = FHI
; Format can be FHI, VASP or CIF
[methods]
cleave_option = 1
[parameters]
layers = 1-9 13 14
; Layers could be specified as combination of start-end or separate numbers by space
vacuum_size = 40
highest_index = 1
; Only needed in cleave_option = 1
supercell_size = None
miller_index = 1 0 0
; Only needed in cleave_option = 0
```


ogreSWAMP.config:


To run ogre
```bash
python run_ogre.py --filename ogre.config
```

To run ogreSWAMP
```bash
python run_ogreSWAMP.py --filename ogreSWAMP.config
```
