# -*- coding: utf-8 -*-

import os
import json
import sys
import numpy as np

from ibslib.io import read,write

class hab_extractor():
    
    def __init__(self, calc_dir, extract_property='get_max_Hab', \
                 property_parent_key=None, log_file=None):
        """
        
        Arguments
        ---------
        calc_dir: str
            Path to a batch Hab calculation directory
        extract_property: str
            String of the function name the user wishes to call
        
        """
        self.calc_dir = calc_dir
        self.extract_property = extract_property
        self.parent_key = property_parent_key
        self.results_dict = {}
        if log_file == None:
            self.log_file = sys.stdout
        else:
            self.log_file = log_file
        
        self.log_file.write('------------------------------------------------\n'+
                            'Performing Hab Extraction\n' + 
                            '------------------------------------------------\n')
    
    
    def run_extractor(self):
        return eval("self._{}()".format(self.extract_property))
    
    
    def _get_max_Hab(self):
        max_Hab_dict = {}
        for struct in os.listdir(self.calc_dir):
            cwd = os.path.join(self.calc_dir,struct,"geometry")
            Hab_results = []
            Hab_results_2 = []
            try:
                f = open(os.path.join(cwd,'fodft_results.json'), "r")
                Hab_results = json.load(f)
                f.close()
            except:
                self.log_file.write('Results in {0} for {1} not found'
                                    .format(cwd,struct))
                pass
            
            cwd = os.path.join(self.calc_dir,struct,"geometry_2")
            try:
                f = open(os.path.join(cwd,'fodft_results.json'), "r")
                Hab_results_2 = np.array(json.load(f))
                f.close()
            except:
                print('Results in {0} for {1} not found'
                      .format(cwd,struct))
            
            Hab_values = []
            if Hab_results != []:
                for member in Hab_results:
                    # Mapping a list to a float
                    # This has to be done because of the seemingly dumb way
                    #    they decided to store the Hab results
                    Hab_values.append(abs(float(member[1][0])))
            if Hab_results_2 != []:
                for member in Hab_results_2:
                    # Mapping a list to a float
                    Hab_values.append(abs(float(member[1][0])))
            
            if Hab_values != []:
                max_val = max(Hab_values)           
                max_Hab_dict[struct] = max_val
        
        return(max_Hab_dict)
    
    
    def _get_dimers(self):
        """
        Saves the dimers from each structure which was calculated as a 
        Structure file. The Hab value from the calculation is stored as a 
        property for each structure. Dimers inherit the name of their parent 
        directory. 
        
        
        """
        struct_dict = {}
        for root, dirs, files in os.walk(self.calc_dir):
            if "aims.out" in files:
                base,name = os.path.split(root)
                # Only want dimer folders
                if "d_" == name[0:2]:
                    dimer_struct = self._dimer_struct(root)
                    Hab = self._Hab_from_aims(root)
                    dimer_struct.set_property("Hab", Hab)
                    struct_dict[dimer_struct.struct_id] = dimer_struct
        
        return struct_dict
                
    
    def _dimer_struct(self, path):
        """
        Converts geometry.in in the dimer calculation folder into a Structure
        object and gives the struct_id based on the input path.s
        
        """
        dimer_struct = read(os.path.join(path,"geometry.in"))
        # Collecting the structure id from  path information
        base_1,dimer_name = os.path.split(path)
        base_2,struct_name = os.path.split(base_1)
        # If parent folder is non-descriptive, use previous folder
        if struct_name == "geometry":
            base_3,struct_name = os.path.split(base_2)
        struct_id = struct_name + "_" + dimer_name
        dimer_struct.struct_id = struct_id
        return dimer_struct
        
        
    def _Hab_from_aims(self,path):
        """
        Takes Hab value from the aims.out file in the path. If there's no 
        aims.out, then it returns 0.
        
        """
        aims_path = os.path.join(path,"aims.out")
        if not os.path.exists(aims_path):
            return 0
        
        with open(aims_path,"r") as f:
            for line in f:
                if "st1 -> st2:     h_ab" in line:
                    # Skip two lines
                    line = next(f)
                    line = next(f)
                    Hab = float(line.split()[3])
                    # Not sure about abs but for now its okay I guess
                    Hab = np.abs(Hab)
                    break
        return Hab
        

if __name__ == '__main__':
    Hab_calc_path = "C:\\Users\\manny\\Research\\Datasets\\Hab_Dimers\\" \
                    "FUQJIK_4mpc_tight\\batchrun"
    
    
    test = hab_extractor(Hab_calc_path, extract_property="get_dimers")
    dimer_dict = test.run_extractor()
    
    write("C:\\Users\\manny\\Research\\Datasets\\Hab_Dimers\\" 
          "FUQJIK_4mpc_tight\\dimers", dimer_dict, overwrite=True)
