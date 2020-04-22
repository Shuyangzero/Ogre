# -*- coding: utf-8 -*-

import torch
import os

from ibslib import Structure,StructDict
from ibslib.io import read,write
from ibslib.acsf.neighborlist import NeighborList,init_cutoffs


class Driver():
    """
    Class that drives ACSF calculations and enables parallel ACSF function 
    calculation over a directory of structures by adhering to the API 
    required by ibsib.libmpi.parallelcalc.
    
    Arguments
    ---------
    acsf: mlcsp.acsf object
        Initialized mlcsp acsf object
    file_format: str
        Can be either "torch" or "struct"/"json". Decides how the ACSF 
        information will be saved. 
    cutoff: float
        Cutoff used for the neighborlist calculation
    prop_name: str
        Property to get from structure for the target value. If the prop_name
        is not specified, then the default behavior is that no targets
        will be collected from the structure's properties.
    mode: str
        Mode will be activated only if the file_format is "struct"/"json". 
        At least for now. Mode is only sesitive to either "struct" or if 
        it's not struct. If it's struct, then an atom-average representation
        is constructed for the structure. If it's not struct, then the 
        representation for every atom is save to the structure's properties.
    
    """
    def __init__(self, acsf, file_format="torch", cutoff=6, prop_name="",
                 mode="struct", output_dir="", overwrite=False):
        self.acsf = acsf
        self.cutoff = cutoff
        self.prop_name = prop_name
        self.features = []
        self.targets = []
        self.struct_id = ""
        self.file_format = file_format
        self.mode = mode
        self.output_dir = output_dir
        self.overwrite=overwrite
        
        
    def calc(self, struct_obj):
        """
        Wrapper function to enable operation for both a single Structure 
        object or an entire Structure Dictionary.
        
        Arguments
        ---------
        struct_obj: ibslib.Structure or Dictionary
            Arbitrary structure object. Either dictionyar or structure.
            
        """
        if type(struct_obj) == dict or type(struct_obj) == StructDict:
            self.calc_dict(struct_obj)
        else:
            self.calc_struct(struct_obj)
            
    
    def calc_dict(self, struct_dict):
        """
        Controls calculation for an entire struct_dict. Additionally, will 
        use self.output_dir and self.overwrite to call self.write function 
        for the user.
        
        """
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct)
            self.write(self.output_dir, overwrite=self.overwrite)
            
    
    def calc_struct(self, struct):
        """
        Calculation for a single structure. Will not call the write function
        for you. 
        
        """
        self.struct = struct
        self.features = []
        self.targets = []
        self.elements = []
        self.struct_id = struct.struct_id
        
        cutoffs = init_cutoffs(struct, radius=self.cutoff)
        nl = NeighborList(cutoffs)
        self.atom_struct_dict = nl.calc_struct(struct, 
                                      return_atom_struct=False)
        self.acsf.forward(struct)
        geo_array = struct.get_geo_array()
        elements = struct.geometry["element"]
        for i,pos in enumerate(geo_array):
            if self.prop_name == "Total_atomic_forces":
                target = torch.tensor(
                          struct.properties["Total_atomic_forces"][i][0:3])[None,:]
            elif len(self.prop_name) > 0:
                target = torch.tensor([struct.properties[self.prop_name]])[None,:]
            acsf = torch.tensor(struct.properties["acsf"][i])[None,:]
            
            self.elements.append(elements[i])
            
            if len(self.features) == 0:
                self.features = acsf
                if len(self.prop_name) > 0:
                    self.targets = target
            else:
                self.features = torch.cat([self.features,acsf])
                if len(self.prop_name) > 0:
                    self.targets = torch.cat([self.targets,target])
        
        # Remove acsf from struct properties created from self.acsf class
        del(struct.properties["acsf"])

        ## Perform mode operation on representation
        if self.mode == "struct":
            self.features = torch.mean(self.features, dim=0)
        else:
            pass
            
        self.struct.properties["RSF"] = self.features.numpy().tolist()
    
    
    def write(self, output_dir, file_format="", overwrite=False):
        """
        Writes all atomic entires for each structure in a features output file
        and a targets output file named with the structure id.
        
        """
        if len(file_format) != 0:
            self.file_format = file_format

        if self.file_format == "torch":
            output_path = os.path.join(output_dir,
                                       "{}.pt".format(self.struct_id))
            
            output_dict = {"features": self.features,
                           "targets": self.targets,
                           "elements": self.elements,
                           "name": self.struct_id}
            torch.save(output_dict, output_path)
            
        elif self.file_format == "struct" or self.file_format == "json":
            # No target property needs to be saved for json saving because
            # the property should already be contained in the structure's
            # properties already.
            
            ## Perform mode operation on representation
            #if self.mode == "struct":
            #    self.features = torch.mean(self.features, dim=0)
            #else:
            #    pass
            
            self.struct.properties["RSF"] = self.features.numpy().tolist()
            temp_dict = {self.struct_id: self.struct}
            write(output_dir, temp_dict, file_format="json", 
                  overwrite=overwrite)
        
        
        
        
