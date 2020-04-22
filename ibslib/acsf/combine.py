# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

combine_modes = ["", "ele", "struct"]





class combine():
    """
    Combines the structure features that are calculated by the acsf calculator
    into singular files for NN training. This function allows three modes 
    of operation depending on the target use.  The function will also add the
    file name for which the data comes from for each of the final entires. This
    will help the user track this information later on if there are issues
    with specific structures or if they would like to monitor the convergence
    of the NN generalization with respect to how many structures/relaxations
    the NN is trained on. 
    
    * "": Default operation combines all atoms across all structures together
    in a basic way.
    
    * "ele": Separates results each element type found in the struct_data_path.
    
    * "struct": Does a simple mean operation across the atomic features that 
    are found for the structure, effectively creating a representation for 
    each structure that is equal to the structure's mean atomic environment.
    This enables simple representation across chemical space for large 
    classes of different structures. 
    
    Arguments
    ---------
    struct_data_path: str
        Path to the directory of torch tensor files to use. 
    output_path: str
        Path to the output directory of combined files.
    mode: str
        One of "", "ele", or "struct". Behavior is described above.
    unqiue_ele: list
        If the unique elements are known, you may provide them. 
    ele_per_struct: list of int
        If the number of each element per structure is known, this may 
        significantly speed up the file combination. This is because the 
        results tensors may be allocated one time without tensor concatenation
        to construct the tensors. The ele_per_struct must be ordered w.r.t.
        the unique_ele list.
    prop_name: str
        Used if using struct_mode. Supply the name of the energy if you would
        like to also parse the energy from the results.
        
    """
    
    def __init__(self, struct_data_path, output_path, mode="",
                 unique_ele=[], ele_per_struct=[], features_dim=-1,
                 prop_name=""):
        if mode not in combine_modes:
            raise Exception("Combine mode {} is not available.".format(mode)+
                    " Please use one of {}.".format(combine_modes))
        
        # General setup
        self.mode = mode
        self.file_list = os.listdir(struct_data_path)
        self.results = {}
        self.features_dim = features_dim
        
        # For ele mode
        self.unique_ele = unique_ele
        self.ele_per_struct = ele_per_struct
        
        # For struct mode
        self.prop_name = prop_name
        
        self.fast_method = False
        ## Need to keep track of last row used if using a fast method 
        self.previous_row = 0
        eval("self.check_fast_{}()".format(self.mode))

        for file_name in self.file_list:
            file_path = os.path.join(struct_data_path, file_name)
            eval("self.cat_{}(file_path)".format(self.mode))
            
        torch.save(self.results, output_path)
            
    
    def cat_ele(self, file_path):
        temp = torch.load(file_path)
        elements = np.array(temp["elements"])
        ele_unique,counts = np.unique(elements, return_counts=True)
        
        # Check if there's already an entry in results for each element
        for ele in ele_unique:
            if not self.results.get(ele):
                self.results[ele] = {}
        
        
        ### Slow and fast methods give the same results.
        for i,ele in enumerate(ele_unique):
            idx = np.where(elements == ele)[0]
#            print(idx)
            ele_features = temp["features"][idx,:]
            ele_targets = temp["targets"][idx,:]
            
            ### Do concatentation if we can't use fast method
            ## Concatentation requires a lot of tensor copying in memory so 
            ## this is really not advisable. But it is flexible.
            if not self.fast_method:
                    
                if not self.results.get(ele):
                    self.results[ele]["features"] = ele_features
                    self.results[ele]["targets"] = ele_targets
                    continue
                
                self.results[ele]["features"] = torch.cat(
                                                (self.results[ele]["features"],
                                                ele_features),
                                                dim=0)
                self.results[ele]["targets"] = torch.cat(
                                                (self.results[ele]["targets"],
                                                ele_targets),
                                                dim=0)
                
                    
            ## Use fast method
            else:
                start_row = self.previous_row[ele]
                num_rows = ele_features.shape[0]
                end_row = start_row + num_rows
                
#                print(self.previous_row, end_row, self.results[ele]["features"].shape, ele_features.shape)
                
                self.results[ele]["features"][start_row:end_row,:] = \
                        ele_features
                self.results[ele]["targets"][start_row:end_row,:] = \
                        ele_targets
                
                ## Now need to update the previous row that was used
                # to the end_row
                self.previous_row[ele] = end_row
                        
    
    
    def check_fast_ele(self):
        """
        Check if we can initialize results tensors from the beginning for the
        ele mode.
        
        """
        ### Checking if we can speed-up the processes using provided arguments
        # to construct the shape of the required tensors first so we don't 
        # have to any concatenation. 
        if len(self.unique_ele) > 0:
            if len(self.unique_ele) == len(self.ele_per_struct) and \
                self.features_dim > 0:
                    
                ## Set fast method to true
                self.fast_method = True
                ## And previous row now needs an entry for each element
                self.previous_row = {}
                
                total_files = len(self.file_list)
                for i,ele in enumerate(self.unique_ele):
                    ## Get shape of tensor
                    num_rows = total_files * self.ele_per_struct[i]
                    num_columns = self.features_dim
                    
                    self.results[ele] = {}
                    self.results[ele]["features"] = torch.zeros(num_rows, 
                                                                num_columns)
                    self.results[ele]["targets"] = torch.zeros(num_rows,3)
                    self.results[ele]["names"] = []
                    
                    self.previous_row[ele] = 0
                    
                    
    def cat_struct(self, file_path):
        """
        Calculates the average atom embeddings.

        """
        temp = torch.load(file_path)
        self.results["names"].append(temp["name"])
        features = temp["features"]
        
        # Target has been added for every atom. Average over them to account 
        # for if the target value is different per atom. 
        if len(self.prop_name) > 0:
            targets = temp[self.prop_name].float()
            targets = torch.mean(targets, dim=0)
            
        struct_features = torch.mean(features, dim=0)
        
        if self.fast_method:
            self.results["features"][self.previous_row] = struct_features
            if len(self.prop_name) > 0:
                self.results[self.prop_name][self.previous_row,0] = targets
            self.previous_row += 1
        else:
            self.results["features"] = torch.cat((self.results["features"],
                                                 features),
                                                 dim=0)
            if len(self.prop_name) > 0:
                self.results[self.prop_name] = torch.cat(
                                                (self.results[self.prop_name],
                                                targets),
                                                dim=0)

    
    def check_fast_struct(self):
        """
        Fast method for struct mode just needs to know the number of 
        dimensions of the feature vector.

        """
        total_files = len(self.file_list)
        if self.features_dim > 0:
            self.fast_method = True
            num_rows = total_files
            num_cols = self.features_dim
            self.results["features"] = torch.zeros(num_rows,
                                                   num_cols)
            if len(self.prop_name) > 0:
                self.results[self.prop_name] = torch.zeros(num_rows,1)
            self.results["names"] = []

    
    def cat_(self, file_path):
        raise Exception("Not implemented")
    
    def check_fast_(self):
        raise Exception("Not implemented")
    
    
    
        
def get_fast_kw_ele(struct_data_dir):
    """
    Will automatically return the correct arguments for the fast methods of 
    combine from a directory of structures for ele mode. 
    
    Arguments
    ---------
    struct_data_dir: str
        Path to a folder containing structure data
    
    """
    fast_arguments = \
        {
            "unique_ele": [],
            "ele_per_struct": [],
            "features_dim": -1,
        }
    file_list = os.listdir(struct_data_dir)
    
    file_path = os.path.join(struct_data_dir, file_list[0])
    temp = torch.load(file_path)    
    unique_ele,counts = np.unique(temp["elements"], return_counts=True)
    features_dim = temp["features"].shape[1]
                      
    fast_arguments["unique_ele"] = unique_ele
    fast_arguments["ele_per_struct"] = counts
    fast_arguments["features_dim"] = features_dim
   
    return fast_arguments
        

def get_fast_kw_struct(struct_data_dir):
    """
    Will automatically return the correct arguments for the fast methods of 
    combine from a directory of structures for struct mode. 

    Arguments
    ---------
    struct_data_dir: str
        Path to a folder containing structure data
    
    """
    fast_arguments = \
        {
            "features_dim": -1
        }
    file_list = os.listdir(struct_data_dir)
    file_path = os.path.join(struct_data_dir, file_list[0])
    temp = torch.load(file_path)    
    features_dim = temp["features"].shape[1]
    fast_arguments["features_dim"] = features_dim

    return fast_arguments