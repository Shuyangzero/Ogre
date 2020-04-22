# -*- coding: utf-8 -*-


import os
import numpy as np
from mpi4py import MPI


class _NaiveParallel():
    """
    Base class for naively parallel operations performed through the file 
    system. 
    
    
    """
    def __init__(self, struct_dir="", comm=None):
        self.struct_dir = struct_dir
        
        if comm == None:
            comm = MPI.MPI.COMM_WORLD
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
    
    def get_files(self, path=""):
        """
        Get the idx range for the rank by respecting the size of the 
        communicator and the number of files to be parallelized over. 
        Then return the filename corresponding to these idx and return.
        
        Arguments
        ---------
        path: str
            Path to get files for. If the length is zero, then the default is 
            to use the variable self.struct_path
            
        """
        if len(path) == 0:
            path = self.struct_path
            
        ## Get all files in the target directory
        file_list = np.array(os.listdir(path))
        
        ## Split files for each rank into most even 
        ## division of work possible
        my_files = file_list[self.rank::self.size]
        my_files = [os.path.join(path,x) for x in my_files]
        
        return my_files


    def get_list(self, arg_list):
        """
        Returns the split of a list for the current rank.

        """
        return arg_list[self.rank::self.size]
    

class _JobParallel():
    """
    Parallelizes naively over a list of jobs to execute.    
    
    """
    def __init__(self, job_list=[], comm=None):
        pass
    


class _LockingParallel():
    """
    Parallelizes over the file system by using Lock files. 
    
    """
    pass
    
    

class _MasterSlave():
    """
    Master and Slave implentation of relatively simple parallelism. In this 
    case, the user may give a list of intructures that need to be executed. 
    
    """
    def __init__(self):
        raise Exception("Not Implemented")
