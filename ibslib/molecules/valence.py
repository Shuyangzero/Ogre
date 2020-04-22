# -*- coding: utf-8 -*-




"""
I would really like an implementation that finds the valence states of 
each atom given the connectivity of a molecule. 

This would have many uses.

"""

symbol_list = \
    ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
     "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
     "Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
     "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm",
     "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
     "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa",
     "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]

valence_electrons_list = \
        [0,1,2,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,2,2,2,1,2,2,2,2,1,2,3,4,5,6,
         7,8,1,2,2,2,1,1,1,1,1,1,2,3,4,5,6,7,8,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
         2,2,2,2,2,2,2,1,1,2,3,4,5,6,7,8,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]

valence_electrons = {}
for idx,value in enumerate(valence_electrons_list):
    valence_electrons[symbol_list[idx]] = value
    

multi_bonds = \
    {
        "B": [2],
        "C": [2,3],
        "N": [2,3],
        "O": [2],
        "P": [2,3],
        "S": [2]
    }


def octet(struct, bond_kw={}, hydrogens=False):
    """
    Calculates whether a closed shell has been satisfied for every atom in the 
    system.
    
    Notes:
        - Need to include oxidation states of transition metals
        - Create scheme for identifying double, triple bonds and resonance 
          states to satisfy octet rules.
        - How to handle systems without hydrogens in a reasonable way? And 
          determine where to put hydrogens? Maybe best handled by having an 
          extra argument. 
    
    Arguments
    ---------
    hydrogens: bool
        If True, will find appropriate atoms to add hydrogen atoms to. 
    
    """
    bond_list = struct.get_bonds(**bond_kw)
    ele = struct.geometry["element"]
    
    ## Compile list of atoms that should be check for the presence of double
    ## or triple bonds. 
    valence_list = []
    check_multi_bonds = []
    for idx,bonds in enumerate(bond_list):
        valence_num = valence_electrons[ele[idx]]
        valence_num += len(bonds)
        valence_list.append(valence_num)
        
        if valence_num != 8:
            check_multi_bonds.append(idx)
    
    ## Now checking the possibility of multiple bonds to satisfy the ele
    for idx in check_multi_bonds:
        
        ## Check if ele can form multiple bonds in the first place
        if ele[idx] not in multi_bonds:
            continue
        
        bonds = bond_list[idx]
        ele_bonds = [ele[x] for x in bonds]
        
        avail = []
        for temp_idx,temp_ele in enumerate(ele_bonds):
            if temp_ele in multi_bonds:
                ## Add unpaired electrons from neighbor
                avail.append(8-valence_list[temp_idx])
        
        
        print(idx, ele[idx], ele_bonds, avail)
        
    
    
    
    
    
    
    

