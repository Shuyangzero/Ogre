
import numpy as np
from ibslib.io import read,write
from ogre.utils import UniquePlanes

struct_dict = read("Examples/")

unique_planes_kw = \
    {
        "index": 2,
        "symprec": 0.1,
        "verbose": True,
        "force_hall_number": 0
    }
    
for struct_id,struct in struct_dict.items():
    atoms = struct.get_ase_atoms()
    up = UniquePlanes(atoms, **unique_planes_kw)
    
    print("{}: {} \n{}".format(struct_id, 
                               len(up.unique_idx),
                               np.vstack(up.unique_idx)))

