

import os
from ibslib.io import read,write,check_dir,check_overwrite



mbd_settings = \
    {
        "xc": "1",
        "mbd_cfdm_dip_cutoff": "100.d0",
        "mbd_supercell_cutoff": "20.d0",
        "mbd_scs_dip_cutoff": "120.0",
        "mbd_scs_vacuum_axis": ".false. .false. .false.",
    }


class MBDBatchCalc():
    """
    Class for setting up MBD calculations.
    """
    def __init__(self, struct_dir, settings=mbd_settings, Slurm=None, 
                calc_dir=""):
        if Slurm == None:
            raise Exception("Must supply initialized ibslib.calculators.slurm.Slurm"+ 
                        " class to ibslib.calculators.mbd.MBDBatchCalc.")
        
        self.Slurm = Slurm
        self.settings = settings
        self.struct_dir = struct_dir
        self.struct_dict = read(struct_dir)

        if len(calc_dir) == 0:
            self.calc_dir = "batch_calc"
        else:
            self.calc_dir = calc_dir
    

    def calc(self, calc_dir="", overwrite=False):
        """
        Arguments
        ---------
        calc_dir: str
            Optional to augment the calc dir in calc method. 
        overwrite: bool
            Overwrite control for the settings file, the gemoetry file,
            and the Slurm Submission Script. 
        """
        if len(calc_dir) != 0:
            self.calc_dir = calc_dir
        
        check_dir(self.calc_dir)

        for struct_id,struct in self.struct_dict.items():
            self.calc_struct(struct, overwrite=overwrite)
    

    def calc_struct(self, struct, calc_dir="",overwrite=False):
        """
        Calculate a single structure in the struct_dict

        Arguments
        ---------
        struct: Structure
            Structure to calculate
        """
        if len(calc_dir) == 0:
            calc_dir = self.calc_dir

        # Make calculation directory
        check_dir(calc_dir)
        # Make structure directory
        struct_id = struct.struct_id
        struct_path = os.path.join(calc_dir, struct_id)
        check_dir(struct_path)

        struct_file_path = os.path.join(struct_path, "geometry")
        write(struct_file_path, struct, file_format="mbd", overwrite=overwrite)

        settings_path = os.path.join(struct_path,"setting.in")
        self._write_settings(settings_path,overwrite)

        # Move to calculation directory
        cwd_temp = os.path.abspath(os.getcwd())
        os.chdir(struct_path)

        self.Slurm.calc("Submit.sh", overwrite=overwrite)

        os.chdir(cwd_temp)
    

    def _write_settings(self,path,overwrite=False):
        check_overwrite(path, overwrite)
        with open(path,'w') as f:
            mbd_str = ""
            for key,value in self.settings.items():
                mbd_str += "{} {}\n".format(key, value)
            f.write(mbd_str)



class MBDExtract():
    """
    Extract results from MBD output and put in Structure file.
    For now, just using a quick implementation. Not the best or most general.
    """
    def __init__(self, struct_dir, calc_dir):
        self.struct_dir = struct_dir
        self.struct_dict = read(struct_dir)
        self.calc_dir = calc_dir
        self._extract()
        self._write()
    

    def _extract(self):
        for struct_id,struct in self.struct_dict.items():
            energy = 0
            calc_path = os.path.join(self.calc_dir, struct_id)
            mbd_file = os.path.join(calc_path,"mbd.out")
            with open(mbd_file,'r') as f:
                for line in f:
                    if "| MBD@rsSCS energy   :" in line:
                        energy = float(line.split()[6])
            struct.set_property("mbd_energy", energy)
    

    def _write(self):
        write(self.struct_dir, self.struct_dict, overwrite=True)
