# -*- coding: utf-8 -*-




"""
Class that handles the orginization of multiple parallel executions to 
perform SIMD type calculations using ibslib calculations. 

Requirements for ibslib calculation classes to use with ParallelCalc:
    - Must have a class.calc_struct method
    - calc_struct method must modify the structure in some way such that the 
      structure can be wrote to storage with the completed calculation 
      results.
    - Module may have an individual function called class.write(otuput_dir)
      which specifies the specific way to write the output for this type 
      of calculation.
"""

import os
import numpy as np

from ibslib.io import read,write
from ibslib.io.write import check_dir
from ibslib.libmpi.base import _NaiveParallel

from mpi4py import MPI


class ParallelCalc(_NaiveParallel):
    def __init__(self, 
                 struct_path, 
                 output_path, 
                 ibslib_class, 
                 file_format="json",
                 overwrite=False,
                 use_class_write=True, 
                 comm=None, 
                 verbose=True):
        """
        Simple implementation that splits work evenly over all ranks.
        
        Arguments
        ---------
        struct_path: str
            Path to the directory of structures to calculate. The path is used
            to avoid loading a copy of the entire directory to every rank.
        ibslib_class: object 
            Class to calculate over every structure in the struct_path. Class
            must be initialized already.
        file_format: str
            File format to use to save the resulting file.
        use_class_write: bool
            If the class.write(output_path) should be used in-place of writing
            the modified structure object. Default is True. However, if the 
            code detects that the class doesn't have a class.write method, 
            it will behave as if the value was False. If False, the structure
            will be writen.
            
        """
        if comm == None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.verbose = verbose
        if self.verbose:
            print("(rank/size): ({}/{})".format(self.rank,self.size))
        
        self.struct_path = struct_path
        self.output_path = output_path
        self.file_format = file_format
        self.overwrite = overwrite
        self.ibslib_class = ibslib_class
        self.use_class_write = use_class_write
        if self.rank == 0:
            check_dir(self.output_path)
        comm.Barrier()
    
    
    def calc(self):
        """
        Wrapper for self.use_class_write
        
        """
        if self.use_class_write:
            try: 
                self.ibslib_class.write
                calc_func = self._calc_write
            except:
                calc_func = self._calc
        else:
            calc_func = self._calc
        
        calc_func()
    
    def _calc(self):
        my_files = self.get_files(self.struct_path)
        total = len(my_files)
        for file_path in my_files:
            struct = read(file_path)
            self.ibslib_class.calc_struct(struct)
            
            temp_dict = {struct.struct_id: struct}
            write(self.output_path, temp_dict, 
                  file_format=self.file_format,
                  overwrite=self.overwrite)
            
            total -= 1
            if self.verbose:
                print("{}: {}".format(self.rank, total))
    
    
    def _calc_write(self):
        """
        Same as _calc but uses calls class.write(output_path) after calculation.
        
        """
        my_files = self.get_files(self.struct_path)
        total = len(my_files)
        for file_path in my_files:
            struct = read(file_path)
            self.ibslib_class.calc_struct(struct)
            self.ibslib_class.write(self.output_path,
                                    file_format=self.file_format,
                                    overwrite=self.overwrite)
            
            total -= 1
            if self.verbose:
                print("{}: {}".format(self.rank, total))
        

if __name__ == "__main__":
    pass

#    from ibslib.motif.classify_motif import MotifClassifier
#    from ibslib.motif.utils import MoleculesByIndex
#    
#    mbi = MoleculesByIndex(13)
#    mc = MotifClassifier(mbi, num_mol=6)
#        
#    pc = ParallelCalc("/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/Property_Based/GAtor_025/100_structures/database_json_files",
#                      "/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/Property_Based/GAtor_025/test_mpi_calc",
#                      mc,
#                      overwrite=True,
#                      )
#    
#    pc.calc()
        
        


#class ParallelCalc():
#    def __init__(self, struct_path, ibslib_class):
#        """
#        Simple implementation which locks files for when structure is being
#        calculated.
#        
#        Arguments
#        ---------
#        struct_path: str
#            Path to the directory of structures to calculate. The path is used
#            to avoid loading a copy of the entire directory to every rank.
#        ibslib_class: object 
#            Class to calculate over every structure in the struct_path
#            
#        """
#        self.struct_path = struct_path
#        self.calc = ibslib_class
#        
#    
#    def calc(self):
#        
    
        
    

##### Master slave rank implementation idea
#class ParallelCalc():
#    def __init__(self, struct_path, ibslib_class):
#        """
#        
#        Arguments
#        ---------
#        struct_path: str
#            Path to the directory of structures to calculate. The path is used
#            to avoid loading a copy of the entire directory to every rank.
#        ibslib_class: object 
#            Class to calculate over every structure in the struct_path
#            
#        """
#        if size == 1:
#            raise Exception("ParallelCalc has been called with on 1 process."+
#                    "ParallelCalc must be called with more than 1 process.")
#            
#        self.struct_path = struct_path
#        self.calc = ibslib_class
#        
#        if rank == 0:
#            self._master_rank()
#        else:
#            self._slave_rank()
#    
#    
#    def _master_rank(self):
#        struct_file_list = os.listdir(self.struct_path)
#        self._master_get_ready_ranks()
#        for file in struct_file_list:
#            file_path = os.path.join(file, self.struct_path)
#    
#    
#    def _master_get_ready_ranks():
#        """
#        Receives all ready ranks.
#        """
#            
#    
#    def _slave_rank(self):
#    
#        
#    
        
        
