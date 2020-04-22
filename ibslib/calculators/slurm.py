

import os
from ibslib.io import check_overwrite
from .slurm_arguments import *



class Slurm():
    """
    Class for the creating of Slurm submission scripts

    Arguments
    ---------
    slurm_arguments: dictionary
        Dictionary with key which specifies the slurm option and the value is
        the intended value. There are two special keys in the argument 
        dictionaries:
            "command": Command which is printed last in the submission script.
            "pre-command": Environment modifiers to be printed before the 
                           command.

    """
    def __init__(self, slurm_arguments):
        if slurm_arguments["command"] == None or \
            len(slurm_arguments["command"]) == 0:
            raise Exception("A valid command was not specified in arguments "+
                    "to the slurm class. Provided arguments were: {}."
                    .format(slurm_arguments))
        self.slurm_arguments = slurm_arguments

    
    def calc(self, file_path, overwrite=False):
        """
        Call write and run together.

        Arguments
        ---------
        file_path: str
            Path to the submission script to be submitted. 

        """
        self.write(file_path,overwrite=overwrite)
        self.submit(file_path)


    def write(self, file_path, overwrite=False):
        """
        Write the submission script.
        """
        # Write all arguments for #SBATCH to file
        file_str = "#!/bin/bash \n"
        for key,value in self.slurm_arguments.items():
            if key == "command" or key == "pre-command":
                continue
            # Check for double dash keywords first
            elif "--" == key[0:2]:
                file_str += "#SBATCH {}={} \n".format(key,value)
            elif "-" == key[0]:
                file_str += "#SBATCH {} {} \n".format(key,value)
            else:
                raise Exception("Slurm object could not hand writing {} {} "
                        .format(key,value) + "to the slurm submission file.")
        
        # Write command to file
        file_str += "\n"
        if self.slurm_arguments["pre-command"] != None:
            file_str += self.slurm_arguments["pre-command"]
            file_str += "\n"
        file_str += self.slurm_arguments["command"]

        # Check if file already exists and if the existing file is the same 
        #   as the generated file_str
        same = self._check_file(file_path, file_str)
        if same == "different":
            check_overwrite(file_path, overwrite=overwrite)
        elif same == "same":
            return
        elif same == "None":
            pass

        with open(file_path, "w") as f:
            f.write(file_str)

        return         


    def submit(self, file_path):
        """
        Submits using sbatch file_path to the queue.
        """
        os.system("sbatch {}".format(file_path))


    def extract_arguments(self,file_path):
        """
        Exctracts the arguments which were used to generate the slurm file. 
        """
        raise Exception("NOT IMPLEMENTED")

    
    def _check_file(self, file_path, file_str):
        """
        Returns whether the file 
        """
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, "r") as f:
                existing_file_str = f.read()
            if existing_file_str == file_str:
                return "same"
            else:
                return "different"
        else:
            return "None"

    

def make_submit(cluster="arjuna", file_name="Submit.sh",
                overwrite=False):
    """
    Function for quickly creating a correct Submit script, likely to be 
    modified by hand after creation.

    """
    args = eval(cluster)
    slurm = Slurm(args)
    slurm.write(file_name, overwrite=overwrite)