# -*- coding: utf-8 -*-


from ibslib import StructDict
from ibslib.io import read,write


class BaseDriver_():
    """
    Base ibslib Driver class that defines the API for all ibslib Drivers. 
    Any Driver should inherit this classes API.
    
    """
    def __init__(self, **settings_kw):
        ## initialize settings
        pass
    
    
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
        Calculates entire Structure dictionary.
        
        """
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct_obj)
            
    
    def calc_struct(self, struct):
        """
        Perform driver calculation on the input Structure. 
        
        """
        driver_info = ["Drivers may modify Structure objects. " +
                       "Because Structures are user defined objects, " +
                       "they are referenced by memory in Python. " + 
                       "This means any modification made to the Structure "+
                       "will be seen outside the function as well."]
        
        struct.properties["DriverInfo"] = driver_info
        
        write_info = ["Although, we may want to save the structure as a part "+
                      "of the class so it can be written as the output by the "+
                      "Driver.write method."]
        struct.properties["WriteInfo"] = write_info
        
        self.struct = struct
            
    
    def write(self, output_dir, file_format="json", overwrite=False):
        """
        Writes the Driver's output to the to the output_dir argument. The only 
        specification of the Driver.write function for the ibslib API are the 
        arguments specified above. The output of a Driver is not necessarily a 
        Structure, although this is the most common situation so the write 
        arguments are tailored for this purpose. In principle, the output of an 
        ibslib Driver is not specified by the API and could be anything.  
        
        Usually, a Driver will only need to output a Structure file, with the 
        struct_id as the name of the output file, to the output directory. 
        This is what's done here.
        
        """
        ## Define dictionary for write API. 
        temp_dict = {self.struct.struct_id: self.struct}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
        