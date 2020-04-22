# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from ibslib.libmpi.base import _NaiveParallel

from mpi4py import MPI


def test_fn(self, dsets, comm=None, **settings):
    """
    
    test_fn has comm argumnet. This means that the user may define any 
    communicator they would like (either all the ranks, a subset of the ranks,
    or one rank at a time). 
    
    Additionally, the Comm can be used in a non-naively parallel way because 
    we have simply recieved them and can do whatever we would like. The Comm
    will be assigned in a naively parallel way, but the Comms need to perform
    naively parallel tasks. However, these tasks are naively parallel w.r.t. the 
    larger system. 
    
    """
    ## Define regressor 
    
    ## Read in all settings
    
    
    ## Fit model
    
    ## Return Result
    pass


def output_fn(results):
    """
    Constructed arguments for now
    
    These arguments should be all that's need for 
    
    """
    pass


def null_fn():
    return


class ParamSearch(_NaiveParallel):
    """
    Parameter searchers are typically performed using a grid search for simple
    machine learning model construction. Such tasks are Naively Parallel and 
    can be performed simultaneously. 
    
    Param Search can later be adapted for Bayesian Optimization.
    
    This class has the following important options:
        1. Takes dsets list of arbitrary length. 
        2. Train idx determines which entires from dset to use as the training
           set. The user may specify more than one idx if they would like to 
           include the train idx as a parameter to search over. 
        3. All other datasets are considered to be testing datasets. 
        4. dset labels may be fed in manually. 
        5. Cross_val determines whether the test_fn will be called multiple 
           times with different testing data splits data splits. 
        6. ranks_per_test: test need not be single threaded and multiple ranks 
        coule be used to perform a single test. For example, I will want to 
        do greedy model construction with respect to the features in the 
        dataset. 
        7. test_fn is a required option because without it, nothing else 
           can work. 
        8. output_fn is also a required function because it is extremely hard
           to predict how a user wants to represent their results. This is 
           due to different tests and models requiring different types of 
           outputs to analyze. For example, decision tree models may want to 
           output the learned tree, but linear regression will want to output
           the weights. In addition, this function may be used for making plots.
        9. Also, due to general nature of implementation, the dsets may be 
           numpy arrays, pandas dataframes, Structure lists, or anything else.
           This is because all aspects of model training, testing, and 
           output of results are fed the dset objects directly. This 
           parallelized wrapper never interacts with the dsets themselves. 
        10. Settings_list: list of lists. Each List contains the values for 
            the parameters the user wants to perform grid search over. 
            All possible combinations of settings will be tested so the scaling
            of the algorithm is poor. Settings may be integers, floats, 
            or bool.
        11. Typically the finalize function will ran to collect all results onto
            rank 0 so that a summary of the calculation can be output. For 
            example, output the best settings. 
            
    Arguments
    ---------
    dset: dict
        Dictionary of dataset(s) to use. Key will will be use as the label for 
        the dataset. 
    train_label:
        Key to use from dset as the training set. 
    settings_list: dict
        Iterable defining the setting parameters for all settings to search 
        over. 
    ranks_per_setting: int
        Number of ranks to give to each setting calculation. 
        NOTE THIS HAS
    cross_val: int
        If zero, no cross validation is used. Otherwise, uses k fold crossval. 
    
    """
    def __init__(self, 
                 test_fn, 
                 output_fn, 
                 finalize_fn=null_fn,
                 dset={}, 
                 train="train",
                 settings={},
                 cross_val=0, 
                 ranks_per_setting=1,
                 comm=None):
        ## MPI Settings
        if comm == None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.ranks_per_setting = ranks_per_setting
        
        if ranks_per_setting != 1:
            raise Exception("Ranks per setting not equal to 1 has not "+
                    "yet been implemented")
        
        
        ## Build settings array
        self.settings = settings
        self.settings_list = self.build_settings()
        
        ## Book-keeping
        self.dset = dset
        self.test_fn = test_fn
        self.output_fn = output_fn
        self.finalize_fn = finalize_fn
        self.cross_val = cross_val
        self.train_label = train
    
    
    def calc(self):
        my_settings = self.get_list(self.settings_list)
        for setting in my_settings:
            result = self.test_fn(self.dset, 
                                  self.train_label, 
                                  self.cross_val, 
                                  *setting)
            self.output_fn(*result)
        self.finalize_fn()
            
    
    def build_settings(self):
        """
        Builds all settings that will be used.
        
        """
        ## Transform everything to numpy array
        for key,value in self.settings.items():
            self.settings[key] = np.array(value)
        
        ## Build settings grid tensor using meshgrid
        grid = np.meshgrid(*self.settings.values()) 
        
        ## Transform to 2D settings array
        num_settings = 1
        for value in grid[0].shape:
            num_settings *= value
        settings_array = np.zeros((num_settings, len(self.settings)), 
                                  dtype=object)
        for idx,entry in enumerate(grid):
            settings_array[:,idx] = entry.ravel()
            
        return settings_array
    
    
    def get_settings_df(self):
        return pd.DataFrame(data=self.settings_list,
                        columns=[x for x in self.settings.keys()])
    
    

if __name__ == "__main__":
    settings = {"alpha": np.arange(0,10,1),
                "alpha2": np.arange(10,100,1),
                "cycles": np.array([True, False])}
    
    grid = np.meshgrid(*settings.values())
    num_settings = grid[0].shape[0]*grid[0].shape[1]*grid[0].shape[2]
    
    settings_array = np.zeros((num_settings, len(settings)), dtype=object)
    for idx,entry in enumerate(grid):
        settings_array[:,idx] = entry.ravel()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        