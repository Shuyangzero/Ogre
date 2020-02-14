

import numpy as np
import spglib as spg


class UniquePlanes():
    """
    Given a ASE atoms crystal structure, finds and returns the unique planes. 
    The algorithm first finds the space group for the system. More specifically,
    it finds the Hall Number which is particularly important for monoclinic 
    systems where it's necessary to know the direction of the unique axis. Then,
    the symmetry operations of the space group are applied to miller index to 
    identify how it transfroms. If the miller index transforms to onto another,
    this these are necessarily specifying identical planes. This information is
    catalogued inside the class and a list of unique miller indices is returned. 
    
    You may also note, symmetry operations with translation components, 
    including centering operations, screw axis, and glide planes create
    additional degenate planes. For example, if the (100) plane is in the
    direction of a 2_1 screw, then the (100) and (200) planes are identical. 
    
    For a compelete list of space groups and hall numbers, please visit:
        http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en
    
    Arguments
    ---------
    atoms: ase.atoms
        Crystal structure to identify unique planes
    index: int
        Maximum miller index to use in plane creation. 
    min_d: float
        Minimum interplanar spacing to use in Angstroms. As distance 
        decreases, the morphological importance decreases. 
    z_prime: float
        Number of molecules in the asymmetric unit. If the number is equal
        to 1, then (100) and (200) will necessarily be the same. However, 
        if there is more than 1 molecule in the asymmetric unit, this 
        is not generally the case. 
    symprec: float
        Precision used within spglib for space group identification. 
    verbose: bool
        True if the user would like informative print statements during
        operation. 
    
    """

    def __init__(self, atoms, index=1, min_d=1.0, z_prime=1,
                 symprec=1e-3, verbose=True):
        if index < 0:
            raise Exception("Index must be greater than zero.")
        if np.sum(atoms.pbc) != 3:
            raise Exception("Atoms object was not a 3D crystal structure.")

        self.atoms = atoms
        self.index = index
        self.symprec = symprec
        self.verbose = verbose

        self.all_idx = self.prep_idx()
        ## Remove zero index
        self.all_idx = self.all_idx[1:, :]

        self.hall_number = self.get_hall_number(atoms, symprec=self.symprec)
        self.find_unique_planes(self.hall_number, z_prime=z_prime)

    def prep_idx(self):
        """
        Prepares all possible miller indices using the maximum index. 
        
        """
        idx_range = np.arange(-self.index, self.index+1)[::-1]
        ## Sort idx_range array so the final list is sorted by magnitude
        ## so that lower index, and positive index, planes are given preference
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        return np.array(
            np.meshgrid(idx_range, idx_range, idx_range)).T.reshape(-1, 3)

    def get_cell(self, atoms):
        """
        Returns a cell tuple of the atoms object for use with spglib
        
        """
        lattice = atoms.cell.tolist()
        positions = atoms.get_scaled_positions().tolist()
        numbers = atoms.numbers.tolist()
        cell = (lattice, positions, numbers)
        return cell

    def get_hall_number(self, atoms, symprec=1e-3):
        """
        Get Hall number using spglib
        
        Arguments
        ---------
        atoms: ase.atoms
            Crystal structure to identify space group
        symprec: float
            Precision used for space group identification. 
        
        """
        cell = self.get_cell(atoms)
        dataset = spg.get_symmetry_dataset(cell,
                                           symprec=symprec)
        if self.verbose:
            print("Space group identified was {}"
                  .format(dataset["international"]))

        return dataset["hall_number"]

    def idx_to_str(self, idx):
        """
        Turns idx array into a unique string representation for use that is
        numerically stable in a hash table. Only two decimal places after the 
        float are only ever required for miller index computations. 
        
        """
        return ",".join(["{:.2f}".format(x) for x in idx])

    def str_to_idx(self, idx_str):
        """
        Turns idx array into a unique string representation for use that is
        numerically stable in a hash table. Only two decimal places after the 
        float are only ever required for miller index computations. 
        
        """
        return [float(x) for x in idx_str.split(",")]

    def find_unique_planes(self, hall_number, z_prime=1):
        """
        From hall number, calculates the unique planes.
        
        Arguments
        ---------
        hall_number: int
            Hall number of space group for unique plane identification. 
            For a compelete list of space groups and hall numbers, please visit:
            http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en
        z_prime: float
            Number of molecules in the asymmetric unit. If the number is equal
            to 1, then (100) and (200) will necessarily be the same. However, 
            if there is more than 1 molecule in the asymmetric unit, this 
            is not generally the case. 
        
        """
        dataset = spg.get_symmetry_from_database(hall_number)
        self.sym_ops = [(r, t) for r, t in zip(dataset['rotations'],
                                               dataset['translations'])]
        sym_ops = self.sym_ops

        ## Dictionary to store idx that have already been used by algorithm
        ## Dictonary is used for O(1) lookup time because of internal hash tabel
        self.used_idx = {}

        ## For use when symmetry elements have translation components
        self.not_used_idx = {}
        self.not_used_idx.update(zip([self.idx_to_str(x) for x in self.all_idx],
                                     [x.tolist() for x in self.all_idx]))

        ## List to compile unique indicies
        self.unique_idx = []

        for idx in self.all_idx:
            ## str format for dictionary storage to be independent of numerical
            ## precision.
            idx_str = ",".join(["{:.2f}".format(x) for x in idx])

            ## First check if idx has been used before
            if self.used_idx.get(idx_str):
                continue
            else:
                ## Otherise it must be a unique index
                self.unique_idx.append(idx)
                del(self.not_used_idx[idx_str])

                ## Doesn't really matter what the value is
                self.used_idx[idx_str] = ". ".join([
                    "Thanks for reading the source code",
                    "Please checkout my website: ibier.io"])

            ## Now apply all symmetry operations to idx and add these to dict
            for rotation, translation in sym_ops:
                transformed = np.dot(rotation, idx) + translation

                trans_str = ",".join(["{:.2f}".format(x) for x in transformed])

                ## Can simply add to dictionary
                self.used_idx[trans_str] = ". ".join([
                    "Thanks for reading the source code",
                    "Please checkout my website: ibier.io"])

                if self.not_used_idx.get(trans_str):
                    del(self.not_used_idx[trans_str])

                ## Compute angle to all other idx values if the number of
                ## molecules in the asymmetric unit is less than or equal to one
                if z_prime <= 1:
                    transformed_norm = transformed / \
                        np.linalg.norm(transformed)
                    not_used_list = [x for x in self.not_used_idx.keys()]
                    for not_used_str in not_used_list:
                        not_used_idx = self.str_to_idx(not_used_str)
                        not_used_norm = np.array(not_used_idx) / \
                            np.linalg.norm(not_used_idx)
                        transformed_norm = transformed / \
                            np.linalg.norm(transformed)

                        ## If the angle of the plane norms are the same,
                        ## then they will be identical planes
                        if np.dot(transformed_norm, not_used_norm) == 1:
                            self.used_idx[not_used_str] = ". ".join([
                                "Thanks for reading the source code",
                                "Please checkout my website: ibier.io"])
                            del(self.not_used_idx[not_used_str])

                ## Now we need to handle the case of translational symmetries,
                ## including the case of lattice translations because the
                ## (200) surface will always be identical to the (100)
                if np.sum(translation) > 0.1:
                    ## The effect of a symmetry with a translation component of
                    ## 0.5 on the miller index will be a multiplication by two.
                    half_idx = np.where(np.abs(translation-0.5) < 0.01)[0]
                    factor = np.ones(3)
                    factor[half_idx] = 2
                    transformed = np.dot(rotation, idx)*factor
                    trans_str = self.idx_to_str(transformed)
                    self.used_idx[trans_str] = ". ".join([
                        "Thanks for reading the source code",
                        "Please checkout my website: ibier.io"])

                    ## For a component of 0.25, this will be a multiplication by
                    ## two and a multiplication by four, so both should be
                    ## considered


if __name__ == "__main__":
    from ibslib.io import read
    s = read("/Users/ibier/Research/Results/Hab_Project/genarris-runs/" +
             "GIYHUR/20191203_Exp_Volume/acsf_report/acsf/raw" +
             "/GIYHUR_2mpc_0afa5b7040.json")
    atoms = s.get_ase_atoms()
    up = UniquePlanes(atoms, 2)
