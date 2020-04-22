# -*- coding: utf-8 -*-

import os,sys,json,copy
import numpy as np

from ibslib import Structure


# List of all implemented properties for extraction from FHI-aims
fhi_properties = \
    [
      "energy",
      "vdw_energy",
      "time", 
      "sg", "space_group", 
      "hirshfeld_volumes", "atom_volumes",
    ]


def extract(struct_dir, kwargs):
    extractor = aims_extractor(struct_dir, **kwargs)
    struct_dict = extractor.run_extraction()
    return struct_dict


def name_from_path(file_path):
    """
    Example naming function. Users can construct their own naming functions
    by taking the file_path as the argument and returning a name for the 
    structure. This naming function uses the last str in the file_path
    for the structure name.

    Arguments
    ---------

    """
    name = os.path.split(file_path)
    if name[-1] == '':
        name = os.path.split(name[0])[-1]
    else:
        name = name[-1]
    return(name)


def name_abs_path(file_path):
    """
    Turns the entire file_path into the name of the structure. 

    """
    # Use normpath for operating system agnostic implementation
    path = os.path.normpath(file_path)
    path = os.path.abspath(path)
    name = file_path.replace("/","_")
    return name


class AimsExtractor():
    """

    Arguments
    ---------
    aims_property: list  of str
        Controls what will be extracted from the FHI-aims calculation.
        Acceptable values are:
            energy: Total energy value
            time: Timing of calculation 
            sg: Also accepts "space_group". Calculates spacegroup with pymatgen
            hirshfeld_volumes: Also accepts "atom_volume". 
                               Collects hirshfeld atom volumes from the output 
                               file and stores them.
    """
    def __init__(self, struct_dir, aims_property=['energy', 'time', "sg"], 
                 energy_name='energy_tier1_relaxed', log_file=None,
                 name_func=name_from_path, symprec=1.0):
        
        self.struct_dir = struct_dir
        self.file_list = os.listdir(struct_dir)
        self.aims_property = aims_property
        self.aims_results_dict = {}
        self.energy_name = energy_name
        self.name_from_path = name_func
        self.symprec = symprec
        
        if log_file == None:
            self.log_file = sys.stdout
        else:
            self.log_file = log_file
            
        self.log_file.write('------------------------------------------------\n'+
                            'Performing Aims Output Extraction\n' + 
                            '------------------------------------------------\n')
        
        
    def run_extraction(self):
        dir_check = self.check_aims_dir()
        if dir_check == 0:
            name = self.name_from_path(self.struct_dir)
            results = self.extract_results(self.struct_dir, name)
            self.aims_results_dict[name] = results
            
        while len(self.file_list) != 0:
            file = self.file_list.pop()
            cwd = os.path.join(self.struct_dir,file)
            dir_check = os.path.isdir(cwd)
            #print(dir_check, file)
            if dir_check == False:
                continue
            else:
                dir_tree = self.create_dir_tree(cwd)

            aims_path = self.find_aims_file(cwd)
            if aims_path != '1':
                name = file
                results = self.extract_results(cwd, name)
                self.aims_results_dict[name] = results
            
            while len(dir_tree) != 0:
                dir_name = dir_tree.pop()
                aims_path = self.find_aims_file(dir_name)
                if aims_path == '1':
                    continue
                name = self.name_from_path(dir_name)
                struct = self.extract_results(dir_name, name)
                self.aims_results_dict[name] = struct                        
                
        return(self.aims_results_dict)
    
    
    def extract_results(self, aims_dir, structure_name):
        """ Extracts specified results to a Structure object
        """
        if 'energy' in self.aims_property or 'time' in self.aims_property:
            aims_path = os.path.join(aims_dir, 'aims.out')
            results = self.extract_from_output(aims_path)
        aims_path = os.path.join(aims_dir, 'aims.out')
        struct = self.make_struct(aims_dir, structure_name)
        for key,value in results.items():
            struct.set_property(key,value)
        return struct
            
        
    def check_aims_dir(self):
        if 'aims.out' in self.file_list:
            pass
        else:
            return(1)
        if 'control.in' in self.file_list:
            pass
        else:
            return(1)
        if 'geometry.in' in self.file_list:
            pass
        else:
            return(1)
        return(0)
    
    
    def find_aims_file(self, cwd):
        aims_name = 'aims.out'    
        try: 
            f = open(os.path.join(cwd, aims_name))
            f.close()
        except:
            self.log_file.write('Could not find aims.out file for {0} \n'\
                           .format(cwd))
            return('1')
               
        aims_path = os.path.join(cwd,aims_name)
        return(aims_path)
    
    
    def create_dir_tree(self, file_path):
        dir_list = []
        for root, dirnames, filenames in os.walk(file_path):
            for dirname in dirnames:
                dir_list.append(os.path.join(root, dirname))
        return(dir_list)
            
        
    def extract_from_output(self, aims_path):
        results = {}
        total_energy = 0
        total_time = 0
        relaxation_steps = 0
        scf_it = 0
        hirshfeld_volumes = []
        free_volumes = []
        vdw_energy = 0
        # Place these in the order they appear in aims.out
#        search_list = [
#                       '| Total energy of the DFT',
#                       '| Number of self-consistency cycles          :',
#                       '| Number of relaxation steps                 :',
#                       '| Total time                                 :',
#                      ]
        
        with open(aims_path,'r') as f:
            for line in f:
                if "Performing Hirshfeld analysis of fragment charges and moments." in line:
                    free_volumes = []
                    hirshfeld_volumes = [] 
                elif "|   Free atom volume        :" in line:
                    free_volumes.append(float(line.split()[5]))
                elif "|   Hirshfeld volume        :" in line:
                    hirshfeld_volumes.append(float(line.split()[4]))
                elif "| vdW energy correction         :" in line:
                    vdw_energy = float(line.split()[7])
                elif '| Total energy of the DFT' in line:
                     total_energy = line.split()[11]
                elif '| Total time                                 :' in line:
                     total_time = line.split()[4]
                elif '| Number of self-consistency cycles          :' in line:
                    scf_it = line.split()[6]
                elif '| Number of relaxation steps                 :' in line:
                    relaxation_steps = line.split()[6]
        
        # Calculating relative hirshfeld volumes
        free_volumes = np.array(free_volumes)
        hirshfeld_volumes = np.array(hirshfeld_volumes)
        relative_hirsfeld_volumes = hirshfeld_volumes / free_volumes
        relative_hirsfeld_volumes = relative_hirsfeld_volumes.tolist()

        for prop in self.aims_property:
            if prop == 'energy':
                results[str(self.energy_name)] = float(total_energy)
            if prop == 'time':
                results['Total Calculation Time'] = float(total_time)
            if prop == 'scf':
                results['SCF Iterations'] = float(scf_it)
            if prop == 'relaxation':
                results['Relaxation Steps'] = float(relaxation_steps)
            if prop == "hirshfeld_volumes" or prop == "atom_volumes":
                results["hirshfeld_volumes"] = relative_hirsfeld_volumes
            if prop == "vdw_energy":
                results["vdw_energy"] = vdw_energy

        return(results)
        
        
    def make_struct(self, cwd, struct_id):
        struct = Structure()
        files = os.listdir(cwd)
        if 'geometry.in.next_step' in files:
            geometry_file = os.path.join(cwd, 'geometry.in.next_step')
        elif 'geometry.in' in files:
            self.log_file.write("Could not find geometry.in.next_step. "+
                        "Using geometry.in for {}\n".format(cwd))
            geometry_file = os.path.join(cwd, 'geometry.in')
        else:
            self.log_file.write('Could not find geometry file for {0} \n'
                                .format(cwd))
            return()
        
        struct.build_geo_from_atom_file(geometry_file)
        struct.struct_id = struct_id
        
        try: 
            struct.get_unit_cell_volume
            a, b, c = struct.get_lattice_magnitudes()
            alpha, beta, gamma = struct.get_lattice_angles()
        except: 
            pass
        else:
            struct.set_property("alpha", alpha)
            struct.set_property("beta", beta)
            struct.set_property("gamma", gamma)
            struct.set_property("a", a)
            struct.set_property("b", b)
            struct.set_property("c", c)
        
        if "space_group" in self.aims_property or "sg" in self.aims_property:
            space_group = self.get_space_group(struct)
            struct.set_property('space_group', space_group)

        return(struct)
    
    
    def get_space_group(self, structure):
        structp = structure.get_pymatgen_structure()
        SG = self.return_spacegroup(structp)
        return(SG)
    
    
    def return_spacegroup(self, structp):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SGA
        try: #newer pymatgen
            spacegroup = SGA(structp, symprec=self.symprec).get_space_group_number()
        except: #older pymatgen
            spacegroup = SGA(structp, symprec=self.symprec).get_spacegroup_number()
        return spacegroup


if __name__ == "__main__":
    pass

#    struct_dir = "/Users/ibier/Research/Results/Hab_Project/FUQJIK/2_mpc/Genarris/Relaxation/batch_eval"
#    results = extract(struct_dir)
#    
#    from ibslib.io import output_struct_dict
#    
#    output_struct_dict("/Users/ibier/Research/Results/Hab_Project/FUQJIK/2_mpc/Genarris/Relaxation/json",
#                       results, file_format="json")
