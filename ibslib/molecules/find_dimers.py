# -*- coding: utf-8 -*-


from ibslib.driver import BaseDriver_
from ibslib.molecules import FindMolecules


class FindDimers(BaseDriver_):
    """
    Given a molecular crystal structure, finds the unique dimers. 
    
    1. Begin by finding molecules in the unit cell according to a FindMolecules
    type of class. 
    2. Make sure that these are whole molecules
    3. Apply combinations of the lattice vectors to the molecules. Create scheme
    to decide how many times to add vectors. 
    4. Take combinations of molecules to generate dimers. 
    5. Test unqiueness of dimers by centering them and then seeing if the 
    systems can be rotated on top of one another. 
    6. Output unique dimers. 
    
    """
    def __init__(self, ):
        pass
    