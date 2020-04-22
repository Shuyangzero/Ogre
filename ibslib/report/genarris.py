# -*- coding: utf-8 -*-

import os,json,shutil,copy

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist 

from ibslib.io import read,write,check_dir
from ibslib.analysis import get
from ibslib.plot.basic import text_plot
from ibslib.plot.ml import pca
from ibslib.plot.genarris import pca_exp
from ibslib.report import Report_, parse_conf

from ibslib.acsf import Driver, RSF, BaseACSF
from ibslib.libmpi import ParallelCalc,Check

from mpi4py import MPI

"""
Develop a base Genarris report class the handles the job 
of attaining all of the relevant paths for anything the user would desire and 
checking which of these paths exists with the relevant structure files. 

This base class can then be used for more complex analysis such as the 
Downsample report, or a basic Genarris report. In addition, it can be 
for the development of any related analysis as the program or analysis 
methods change.

Develop such a class for GAtor. 

"""


class DownsampleReport(Report_):
    """
    Creates a report to visualize the downsampling of Genarris using a 
    molecular packing descriptor and PCA. Supports different modes of 
    operation. 
        1. Individual: Perform PCA for each portion of the downsampling process. 
           The PCA plots will not have any relation to one another. 
        2. AP: PCA vectors will be calculated using the raw structures. These PCA
           vectors will be kept for the downsampling process but not for 
           the relaxed structures. This is because the unrelaxed structures
           may exhibit packing patterns that are not present in relaxed 
           structures. 
        3. Raw: PCA vectors will be calculated using the raw and will be used 
           for everything.
        4. Relaxed: PCA vectors will be calculated using the relaxed structures 
           and used for everything. 
        5. All: PCA vectors will be calculated by combining the raw structures
           and relaxed structures and used for everything. 
           
    These modes of operation give the user the ability to draw correlations
    between the downsampling process and potentially between the unrelaxed
    and relaxed structures. 
    
    This report supports parallel operation due to support for computing the 
    ACSF descriptors.
    
    Additionally, the report supports interacting with a configuration file
    or with user defined inputs.
    
    Lastly, using the functions such as set_struct_path, check_acsf, and 
    calc_acsf as templates, different downsample workflows could be defined. 
    It would take a littler work to implement a new workflow report, but 
    the groundwork has already been done. 
    
    
    Arguments
    ---------
    raw_pool_path: str
        Path to the raw pool of structures
    ap1_path: str
        Path to the structures after the first round of clustering
    ap2_path: str
        Path to the structures after the second round of clustering
    relaxed_path: str
        Path to the structures after relaxation.
    rsf_path_raw: str
        Path to the directory containing structures with ACSF calculated 
        for the raw pool.
    rsf_path_relaxed: str
        Path to the directory containing structures with ACSF calculated 
        for the relaxed structures.
    close_n_structs: int
        Finding the top n closest structures to the experimental within the 
        projections. 
    
    """
    def __init__(self, 
                 conf_path="", 
                 json_path="", 
                 acsf_key="RSF",
                 report_name="report.pdf",
                 report_folder="acsf_report", 
                 report_width=11,
                 comm=None,
                 raw_pool_path="",
                 ap1_path="",
                 ap2_path="",
                 relaxed_path="",
                 rsf_path_raw="",
                 rsf_path_relaxed="",
                 exp_path="",
                 RSF_kw = \
                    {
                        "struct_dict": {},
                        "cutoff": 12,
                        "unique_ele": [], 
                        "force": False,
                        "n_D_inter": 12, 
                        "init_scheme": "shifted",
                        "eta_range": [0.05,0.5], 
                        "Rs_range": [1,10],
                        "del_neighbor": True
                    },
                driver_kw = \
                    {
                        "file_format": "struct",
                        "cutoff": 12,
                        "prop_name": "",
                    },
                mode="Raw",
                close_n_structs=10,
                use_remake=False, 
                ):
        ## Copy arguments
        arguments = locals()
        arguments_copy = {}
        for key,value in arguments.items():
            ## Can't copy comm so don't use it
            if key == "comm":
                arguments_copy[key] = None
            elif key == "self":
                continue
            else:
                arguments_copy[key] = copy.deepcopy(value)
        arguments = arguments_copy
        
        ## All possible modes
        self.all_modes = ["Individual", "AP", "Raw", "Relaxed"]
        
        ## Prepare and add arguments to the report_conf for reproducibility
        self.report_conf = {}
        self.report_conf["arguments"] = arguments
        
        if comm == None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
            
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
        # Save path the report file will be saved at for future use
        self.report_folder = report_folder
        self.report_name = report_name
        self.remake_path = os.path.join(report_folder,"remake.json")
        self.cwd = os.path.abspath(os.getcwd())
        self.remake = use_remake
        if self.rank == 0:
            check_dir(self.report_folder)
        self.comm.Barrier()
        self.report_path = os.path.join(report_folder, report_name)
        self.report_json_path = os.path.join(report_folder,
                                             "settings.json")
        
        # Setup the configuration dictionary that describes the settings the 
        # Genarris calculation was conducted with.
        self.conf_dict = {}
        if len(conf_path) > 0:
            self.conf_dict = parse_conf(conf_path)
        elif len(json_path) > 0:
            self.parse_json(json_path)
        
        # Initialize information for the actual figure of the report
        self.figure_kw = \
            {
              # Use standard paper width
              "figsize": [report_width,0],
              "constrained_layout": True,
            }
        self.gridspec_kw = \
            {
              "nrows": 0,
              "ncols": 1,
            }
        self.info_textbox_kw = {
                "text_loc": [0.025,0.5],
                "height": 1,
                "edgecolor": "w",
                "text_kw": 
                    {
                        "horizontalalignment": "left",
                        "fontsize": 12,
                        "wrap": True,
                    }}
            
        ## Storage fo later operations
        self.section_calls = []
        self.acsf_key = acsf_key
        self.acsf_calc_dir = os.path.join(self.report_folder, "acsf")
        self.RSF_kw = RSF_kw
        self.driver_kw = driver_kw
        
        ## Add Basic Header
        self.add_header("ACSF Downsampling Report")
        self.add_textbox("Mode: {}"
                         .format(self.report_conf["arguments"]["mode"]),
                         height=1)
            
        ## Organize StructDict storage for each stage of the downsampling 
        ## Note: If these are unpopulated by then end of report creation, 
        ## then these sections will not be added to the report.
        self.raw_pool = {}
        self.ap1 = {}
        self.ap2 = {}
        self.relaxed = {}
        
        ## Saving paths from arguments
        ## In the future, these should probably not exist because it would
        ## be better to interact through self.report_conf["arguments"]
        self.raw_pool_path = raw_pool_path
        self.ap1_path = ap1_path
        self.ap2_path = ap2_path
        self.relaxed_path = relaxed_path
        self.rsf_path_raw = rsf_path_raw
        self.rsf_path_relaxed = rsf_path_relaxed
        self.exp_path = exp_path
    
    
    def _report(self):
        """
        Main calculation steps of the Report.
        
        """
        
        ## Populated all paths
        self.add_header("Path Information")
        self.set_struct_path(self.raw_pool_path, 
                             "Raw Pool",
                             "raw_pool_path",
                             "pygenarris_structure_generation",
                             "output_dir"
                             )
        ## Check if ACSF path for raw pool was defined
        self.check_acsf_path_raw()
        
        self.set_struct_path(self.ap1_path, 
                             "AP-1",
                             "ap1_path",
                             "affinity_propagation_fixed_clusters",
                             "output_dir_2"
                             )
        self.set_struct_path(self.ap2_path, 
                             "AP-2",
                             "ap2_path",
                             "affinity_propagation_fixed_clusters",
                             "exemplars_output_dir_2"
                             )
        self.set_struct_path(self.relaxed_path, 
                             "Relaxation",
                             "relaxed_path",
                             "run_fhi_aims_batch",
                             "output_dir"
                             )        
        ## Check if ACSF path for relaxed pool was defined
        self.check_acsf_path_relaxed()
        
        self.set_struct_path(self.exp_path, 
                     "Experimental",
                     "exp_path",
                     "",
                     "")
        ## Put experimental structure in acsf folder if the file exists
        if self.rank == 0:
            check_dir(os.path.join(self.acsf_calc_dir, "experimental"))
            if len(self.report_conf["arguments"]["exp_path"]) > 0:
                path = self.report_conf["arguments"]["exp_path"]
                exp = read(path)
                dst = "{}/{}".format(self.acsf_calc_dir, 
                                    "experimental")
                check_dir(dst)
                
                if type(exp) != dict: 
                    temp_dict = {exp.struct_id: exp}
                else:
                    temp_dict = exp
                    
                write(dst, temp_dict, overwrite=True)
            
        
        ## Get list of structure names downsampled from raw pool. 
        self.add_header("AP Structure Information")
        self.get_ap()
        
        
        ## Now check what still needs to be calculated and setup directories
        ## for calculations using ibslib.libmpi.Check
        self.add_header("ACSF Calculation Information")
        self.get_unique_ele()
        self.check_acsf(self.report_conf["arguments"]["raw_pool_path"],
                        self.report_conf["arguments"]["rsf_path_raw"],
                        "raw")
        self.check_acsf(self.report_conf["arguments"]["relaxed_path"],
                        self.report_conf["arguments"]["rsf_path_relaxed"],
                        "relaxed")
        self.check_acsf("{}/{}".format(self.acsf_calc_dir,
                                       "experimental"),
                        "",
                        "experimental")
        
        ## Write report before long acsf calculations
        if self.rank == 0:
            self.report(tag=False)
        
        ## Perform calculations by attaching ACSF driver to ibslib.libmpi.Check
        self.calc_acsf(self.report_conf["arguments"]["relaxed_path"],
                       self.report_conf["arguments"]["rsf_path_relaxed"],
                       "relaxed")
        self.calc_acsf(self.report_conf["arguments"]["raw_pool_path"],
                       self.report_conf["arguments"]["rsf_path_raw"],
                       "raw")
        self.calc_acsf("{}/{}".format(self.acsf_calc_dir,
                                      "experimental"),
                       "",
                       "experimental")
        
        
        ## Overwrite the acsf/structure folders in report_conf 
        ## so in the future we can just reference these to recreate the report
        if len(self.report_conf["arguments"]["raw_pool_path"]) > 0:
            self.report_conf["arguments"]["raw_pool_path"] = \
                    "{}/{}".format(self.acsf_calc_dir, "raw")
        if len(self.report_conf["arguments"]["rsf_path_raw"]) > 0:
            self.report_conf["arguments"]["rsf_path_raw"] = \
                "{}/{}".format(self.acsf_calc_dir, "raw")
        if len(self.report_conf["arguments"]["relaxed_path"]) > 0:
            self.report_conf["arguments"]["relaxed_path"] = \
                "{}/{}".format(self.acsf_calc_dir, "relaxed")
        if len(self.report_conf["arguments"]["rsf_path_relaxed"]) >= 0 and \
            len(self.report_conf["arguments"]["relaxed_path"]) > 0:
            self.report_conf["arguments"]["rsf_path_relaxed"] = \
                "{}/{}".format(self.acsf_calc_dir, "relaxed")
        self.report_conf["arguments"]["exp_path"] = \
                "{}/{}".format(self.acsf_calc_dir, "experimental")
        
                
        ## Additionally, the arguments for RSF may need to be modified
        if len(self.report_conf["arguments"]["RSF_kw"]["struct_dict"]) > 0:
            ## Remove struct_dict and only keep unique elements
            self.report_conf["arguments"]["RSF_kw"]["struct_dict"] = {}
            temp_driver = self.get_acsf_calc()
            self.report_conf["arguments"]["RSF_kw"]["unique_ele"] = \
                    temp_driver.acsf.unique_ele.tolist()
        
        
        ## After all of this setup, we are finally ready to calculate 
        ## projections of the ACSF and plot the results. Only 1 rank needs 
        ## to do these steps. 
        if self.comm.rank == 0:
            self.add_header("ACSF Plots")
            self.plot_pca()
            
        ## Overwrite the acsf/structure folders in report_conf 
        ## so in the future we can just reference these to recreate the report
        if len(self.report_conf["arguments"]["raw_pool_path"]) > 0:
            self.report_conf["arguments"]["raw_pool_path"] = \
                "{}/{}".format("acsf", "raw")
        if len(self.report_conf["arguments"]["rsf_path_raw"]) > 0:
            self.report_conf["arguments"]["rsf_path_raw"] = \
                "{}/{}".format("acsf", "raw")
        if len(self.report_conf["arguments"]["relaxed_path"]) > 0:
            self.report_conf["arguments"]["relaxed_path"] = \
                "{}/{}".format("acsf", "relaxed")
        if len(self.report_conf["arguments"]["rsf_path_relaxed"]) >= 0 and \
            len(self.report_conf["arguments"]["relaxed_path"]) > 0:
            self.report_conf["arguments"]["rsf_path_relaxed"] = \
                "{}/{}".format("acsf", "relaxed")
        self.report_conf["arguments"]["exp_path"] = \
                "{}/{}".format("acsf", "experimental")
            
    
    def get_unique_ele(self):
        """
        Finds unique elements from the Relaxed pool. If Relaxed pool is not used
        then tries AP2, AP1, and then Raw pool. This is in order of increasing 
        cost to get the unique elements for the pool of structures. 
        
        """
        if len(self.report_conf["arguments"]["RSF_kw"]["unique_ele"]) > 0:
            return
        
        if self.rank == 0:
            if len(self.report_conf["arguments"]["relaxed_path"]) > 0:
                s = read(self.report_conf["arguments"]["relaxed_path"])
            elif len(self.report_conf["arguments"]["ap2_path"]) > 0:
                s = read(self.report_conf["arguments"]["ap2_path"])
            elif len(self.report_conf["arguments"]["ap1_path"]) > 0:
                s = read(self.report_conf["arguments"]["ap2_path"])
            elif len(self.report_conf["arguments"]["raw_pool_path"]) > 0:
                s = read(self.report_conf["argumnets"]["raw_pool_path"])
            else:
                raise Exception("No structure paths were found.")
            
            ## Juse BaseACSF to get unique ele in correct format
            temp = BaseACSF(s, 0)
            unique_ele = list(temp.unique_ele)
            
            self.report_conf["arguments"]["RSF_kw"]["unique_ele"] = unique_ele
        else:
            unique_ele = []

        self.report_conf["arguments"]["RSF_kw"]["unique_ele"] = \
            self.comm.bcast(unique_ele, root=0)

    
    def get_ap(self):
        """
        Gets the name of structures from the AP downsampling steps. In addition,
        will also get the energy values for structures from the SCF calculations
        if the values can be found. 
        
        This data will be saved and referenced latter from the acsf/raw folder. 
        This is because the downsampling is always with reference to the 
        unrelaxed structures.
        
        """
        
        ## Check if we need to get ids again for ap1
        get_ap1 = True
        if self.report_conf.get("AP1"):
            if len(self.report_conf["AP1"]["struct_id"]) != 0:
                get_ap1 = False
                num_structs = len(self.report_conf["AP1"]["struct_id"])
                temp_str = ["Found {} structure IDs for AP step 1. "
                    .format(num_structs)+
                    "These IDs were obtained previously."]
                self.add_textbox(temp_str[0], **self.info_textbox_kw)
                
        if get_ap1: 
            self.report_conf["AP1"] = {}
            self.report_conf["AP1"]["struct_id"] = []
            if len(self.report_conf["arguments"]["ap1_path"]) > 0:
                ## Can assume this path already exists because it has been 
                ## passed through set_struct_path
                s = read(self.report_conf["arguments"]["ap1_path"])
                id_list = [x for x in s.keys()]
                self.report_conf["AP1"]["struct_id"] = id_list

                ## Now get energies
                results = get(s, "prop", ["energy"])
                self.report_conf["AP1"]["energy"] = \
                     results["energy"].values.tolist()

                num_structs = len(self.report_conf["AP1"]["struct_id"])
                temp_str = ["Found {} structure IDs for AP step 1. "
                    .format(num_structs)+
                    "These IDs were obtained from {}."
                    .format(self.report_conf["arguments"]["ap1_path"])]
                self.add_textbox(temp_str[0], **self.info_textbox_kw)
                
                
                
        
        ## Check if we need to get ids again for ap2
        get_ap2 = True
        if self.report_conf.get("AP2"):
            if len(self.report_conf["AP2"]["struct_id"]) != 0:
                get_ap2 = False
                num_structs = len(self.report_conf["AP2"]["struct_id"])
                temp_str = ["Found {} structure IDs for AP step 2. "
                    .format(num_structs)+
                    "These IDs were obtained previously."]
                self.add_textbox(temp_str[0], **self.info_textbox_kw)
        
        if get_ap2: 
            self.report_conf["AP2"] = {}
            self.report_conf["AP2"]["struct_id"] = []
            self.report_conf["AP2"]["energy"] = []
            if len(self.report_conf["arguments"]["ap2_path"]) > 0:
                s = read(self.report_conf["arguments"]["ap2_path"])
                id_list = [x for x in s.keys()]
                self.report_conf["AP2"]["struct_id"] = id_list
                
                ## Now get energies
                results = get(s, "prop", ["energy"])
                self.report_conf["AP2"]["energy"] = \
                     results["energy"].values.tolist()
                     
                num_structs = len(self.report_conf["AP2"]["struct_id"])
                temp_str = ["Found {} structure IDs for AP step 2. "
                    .format(num_structs)+
                    "These IDs were obtained from {}."
                    .format(self.report_conf["arguments"]["ap2_path"])]
                self.add_textbox(temp_str[0], **self.info_textbox_kw)
                
    
    def report(self, figname="", tag=True, savefig_kw={"dpi": 600}):
        """
        Creates report folder with report, the settings file, and a script to
        remake the report from the settings file. 
        
        """
        self._report()
        
        ### Check if remake file exists in report folder
        if self.remake:
            remake_path = os.path.join(self.report_folder, 
                                       "remake.json")
            if os.path.exists(remake_path):
                self.load_remake()
                
        ## Add ibslib tag. User can easily turn this off if desired. 
        if tag == True:
            self.add_tag()
                    
        self._generate_report(figname=figname, savefig_kw=savefig_kw)
        self.write_make_report()
        self.write_remake()
        self.write_json()
        
    
    def write_make_report(self):
        """
        At the end of the calculation, create an extra file called 
        make_report.py in the self.report_folder. This file will use the 
        newly created settings.json file for report creating. This will 
        help the user recreate reports from only the settings.json file. 
        
        """
        path = os.path.join(self.report_folder, "make_report.py")
        with open(path,"w") as f:
            text = \
                [
                    "from ibslib.report.genarris import DownsampleReport\n" +
                    "json_path = \"settings.json\"\n" +
                    "rp = DownsampleReport(json_path=json_path,report_folder=\"./\", "+
                    "use_remake=False,"+
                    "report_width={})\n".format(self.figure_kw["figsize"][0])+
                    "rp.report()"
                ]
            text = text[0]
            f.write(text)
        
        path = os.path.join(self.report_folder, "same_axis_plot.py")
        with open(path,"w") as f:
            text = ["import json\n" +
                    "from ibslib.plot.genarris import pca_exp\n" + 
                    "from ibslib.report import Report_\n"+
                    "r = Report_(report_name='Same_Axis_Plots.pdf',\n" +
                    "            report_folder='./',\n"+
                    "            use_remake=False,\n"+
                    "            report_width=11)\n"+
                    "\n"+
                    'with open("settings.json") as f:\n'+
                    "    settings = json.load(f)\n"+
                    "\n"
                    'plot_dict = settings["Report Configuration"]["Plot"]\n'+
                    'xlabel_kw = plot_dict["Raw"]["xlabel_kw"]\n'+
                    'ylabel_kw = plot_dict["Raw"]["ylabel_kw"]\n'+
                    'xticks = plot_dict["Raw"]["xticks"]\n'+
                    'yticks = plot_dict["Raw"]["yticks"]\n'+
                    "\n"+
                    'for name in plot_dict:\n'+
                    '    plot_dict[name]["xlabel_kw"] = xlabel_kw\n'+
                    '    plot_dict[name]["ylabel_kw"] = ylabel_kw\n'+
                    '    plot_dict[name]["xticks"] = xticks\n'+
                    '    plot_dict[name]["yticks"] = yticks\n'+
                    '    r.centered_plot("pca_exp(**{}, ax=ax)".format(plot_dict[name]), label=name)\n'+
                    "\n"
                    'r.report(figname="Same_Axis_Plots.pdf", write_remake=False, write_json=False)\n']
            
            text = text[0]
            f.write(text)
            
    
    def calc_acsf(self, struct_path, acsf_path, name):
        """
        Perform ACSF calculations if needed. 
        
        """
        if len(acsf_path) == 0 and len(struct_path) == 0:
            return
            
        output_dir = "{}/{}".format(self.acsf_calc_dir, name)
        calc_dir = "{}/{}_temp".format(self.acsf_calc_dir, name)
        
        num_files = len(os.listdir(calc_dir))
        temp_str = ["Performing {} calculation on {} structures from "
                    .format(self.acsf_key, num_files)+
                    "the {} pool.".format(name)]
        
        self.add_textbox(**{
                "text": temp_str[0],
                "text_loc": [0.025,0.5],
                "height": 1,
                "edgecolor": "w",
                "text_kw": 
                    {
                        "horizontalalignment": "left",
                        "fontsize": 12,
                        "wrap": True,
                    }}
                )
        
        check_kw = \
                {
                    "struct_dir": struct_path,
                    "output_dir": output_dir,
                    "calc_dir": calc_dir,
                    "comm": self.comm,
                    "calculator": self.get_acsf_calc()
                }
        c = Check(**check_kw)
        c.calc()
        self.comm.Barrier()
        
        temp_str = ["Completed Calculations Successfully."]
        self.add_textbox(**{
                "text": temp_str[0],
                "text_loc": [0.1,0.5],
                "height": 1,
                "edgecolor": "w",
                "text_kw": 
                    {
                        "horizontalalignment": "left",
                        "fontsize": 12,
                        "wrap": True,
                    }}
                )
            
        ## Output report after a potentially long calculation
        if self.rank == 0:
            shutil.rmtree(calc_dir)
        
        return
        
        
        
    
    def check_acsf(self, struct_path, acsf_path, name):
        """
        Checks if ACSF calculations need to be performed for input directory 
        of structures. 
        
        Arguments
        ---------
        struct_path: str
            Path to the structures that all need ACSF calculated. 
        acsf_path: str
            Path to the folder of structures with ACSF already calculated. 
        name: str
            Name of the pool of structures ACSF is being checked for.
        
        """
        output_dir = "{}/{}".format(self.acsf_calc_dir, name)
        calc_dir = "{}/{}_temp".format(self.acsf_calc_dir, name)
        ## ACSF path has been set and needs to be checked for already completed
        ## structures
        if len(acsf_path) > 0:
            ## Use check to begin moving completed structures to ACSF report
            ## directory.
            check_kw = \
                {
                    "struct_dir": acsf_path,
                    "output_dir": output_dir,
                    "calc_dir": calc_dir,
                    "comm": self.comm
                }
            c = Check(**check_kw)
            c.calc()
            self.comm.Barrier()
        
        if len(struct_path) > 0:
            ## Use check to finalize calculation directory with any structures 
            ## that have not yet been completed previously.
            ## This is done by using the same output_dir and calc_dir as for 
            ## the acsf_path. 
            check_kw = \
                {
                    "struct_dir": struct_path,
                    "output_dir": output_dir,
                    "calc_dir": calc_dir,
                    "comm": self.comm
                }
            c = Check(**check_kw)
            c.calc()
            self.comm.Barrier()
            
        
        if self.rank == 0:
            ## Check if we will do ACSF at all
            if len(acsf_path) == 0 and len(struct_path) == 0:
                temp_str = ["No calculation for {}.".format(name)]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.025,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
                return
            
            num_struct = len(os.listdir(struct_path))
            num_calc = len(os.listdir(calc_dir))
            temp_str = ["Found {} {} pool structures. {} structures need {} "
                        .format(num_struct, name, num_calc, self.acsf_key) +
                        "to be calculated."]
            self.add_textbox(**{
                    "text": temp_str[0],
                    "text_loc": [0.025,0.5],
                    "height": 1,
                    "edgecolor": "w",
                    "text_kw": 
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                        }}
                    )
            return
        
    
    def check_acsf_path_relaxed(self):
        """
        Checks if the ACSF folder exists for the relaxed pool anywhere in 
        possible arguments. There's no section for ACSF in Genarris conf 
        so there's no reason to check this. 
        
        """
        if len(self.rsf_path_relaxed) > 0:
            pass
        elif self.report_conf["arguments"].get("rsf_path_relaxed"):
            self.rsf_path_relaxed = self.report_conf["arguments"]["rsf_path_relaxed"]
        else:
            self.rsf_path_relaxed = ""
        
        ## Save in report dict
        self.report_conf["arguments"]["rsf_path_relaxed"] = self.rsf_path_relaxed
        
        if len(self.rsf_path_relaxed) > 0:
            if os.path.exists(self.rsf_path_raw):
                temp_str = ["RSF path for relaxed pool found: {}"
                            .format(self.rsf_path_raw)]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
            else:
                temp_str = ["RSF path for relaxed pool not found: {}"
                            .format(self.rsf_path_raw) + 
                            "ACSF calculations will be performed by Report."]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
        else:
            if len(self.report_conf["arguments"]["relaxed_path"]) > 0:
                temp_str = ["RSF path for relaxed pool was not found. " +
                            "Defaulting to relaxed pool path: {}."
                            .format(self.report_conf["arguments"]
                                                    ["relaxed_path"])]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
                
                ## Save rsf_path_raw as RSF path
                self.report_conf["arguments"]["rsf_path_relaxed"] = self.relaxed_path
                self.rsf_path_relaxed = self.relaxed_path
            else:
                temp_str = ["No relaxed pool path found found in arguments."+
                            " No ACSF calculations for the relaxed pool "+
                            "will be performed by Report."]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
    
    
    def check_acsf_path_raw(self, 
                      conf_dict_section="run_rdf_calc",
                      conf_dict_key="output_dir"):
        """
        Checks if ACSF folder exists for raw pool path defined anywhere in 
        the possible arguments. 
        
        """
        ## First check path from instantiation
        if len(self.rsf_path_raw) > 0:
            pass
        elif self.report_conf["arguments"].get("rsf_path_raw"):
            self.rsf_path_raw = self.report_conf["arguments"]["rsf_path_raw"]
        elif self.conf_dict.get(conf_dict_section):
            if self.conf_dict[conf_dict_section].get(conf_dict_key):
                self.rsf_path_raw = self.conf_dict[conf_dict_section][conf_dict_key]
        elif self.conf_dict.get("run_rsf_calc"):
            if self.conf_dict["run_rsf_calc"].get(conf_dict_key):
                self.rsf_path_raw = self.conf_dict["run_rsf_calc"][conf_dict_key]
        else:
            self.rsf_path_raw = ""
        
        ## Save to report_dict
        self.report_conf["arguments"]["rsf_path_raw"] = self.rsf_path_raw
        
        ## Add results to report
        if len(self.rsf_path_raw) > 0:
            ## Check if path exists
            if os.path.exists(self.rsf_path_raw):
                temp_str = ["RSF path for raw pool found: {}"
                            .format(self.rsf_path_raw)]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
            else:
                temp_str = ["RSF path for raw pool not found: {}"
                            .format(self.rsf_path_raw) + 
                            "ACSF calculations will be performed by Report."]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
        else:
            if len(self.report_conf["arguments"]["raw_pool_path"]) > 0:
                temp_str = ["RSF path for raw pool was not found. " +
                    "Defaulting to raw pool path: {}."
                    .format(self.report_conf["arguments"]["raw_pool_path"])]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
                
                ## Save rsf_path_raw as RSF path
                self.report_conf["arguments"]["rsf_path_raw"] = self.raw_pool_path
                self.rsf_path_raw = self.raw_pool_path
            else:
                temp_str = ["No raw pool path found in arguments."+
                            " No ACSF calculations for the raw pool "+
                            "will be performed by Report."]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.1,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
        
    
    def parse_json(self, json_path):
        with open(json_path) as f:
            temp_dict = json.load(f)
        self.conf_dict = temp_dict["Genarris Configuration"]
        self.report_conf = temp_dict["Report Configuration"]
        
    
    def write_json(self, json_path=""):
        """
        Output a json file that has all agruments to regenerate the exact
        same report, without reference to any of the original files. 
        
        """
        temp_dict = {}
        temp_dict["Genarris Configuration"] = self.conf_dict
        temp_dict["Report Configuration"] = self.report_conf
        
        if len(json_path) == 0:
            json_path = self.report_json_path
         
        with open(json_path,"w") as f:
            ## Test to make sure everything is json serializable
            keys = [x for x in temp_dict.keys()]
            for key in keys:
                sub_key_list = [x for x in temp_dict[key].keys()]
                for sub_key in sub_key_list:
                    try:
                        _ = json.dumps(temp_dict[key][sub_key])
                    except:
                        print("{},{} is not json serializable: {}"
                      .format(key,sub_key,temp_dict[key][sub_key]))
                        del(temp_dict[key][sub_key])

        with open(json_path,"w") as f:
            f.write(json.dumps(temp_dict))
                    
    
    def set_struct_path(self, path_var, name, report_conf_name, 
                        conf_dict_section, conf_dict_name):
        """
        Gets the path to the structures in an ordered way starting by checking
        if the path is already set, then checking the report_conf, finally
        by checking the conf_dict. If no path was found, then it sets the 
        value to a string of length zero and no analysis will be performed for
        that section.
        This function is written more generally so that or reuse modification 
        is easier in the future. 
        
        Arguments
        ---------
        path_var: str
            The variable that should be set with the path that is found. 
        name: str
            Plain text name of the path that is being set. For example, 
            raw_pool would be "raw pool"
        report_conf_name: str
            Key to look in the report_conf for the path. 
        conf_dict_section: str
            Section in the conf_dict the path is located. 
        conf_dict_name: str
            Key within the conf_dict section to look for the path. 
        
        
        """
        
        ## First check if path is already set. Will not overwrite.
        if len(path_var) > 0:
            pass
        ## Then check path from json file
        elif self.report_conf["arguments"].get(report_conf_name):
            path_var = self.report_conf["arguments"][report_conf_name]
        ## Check the conf dict section that value name
        elif self.conf_dict.get(conf_dict_section):
            if self.conf_dict[conf_dict_section].get(conf_dict_name):
                path_var = self.conf_dict[conf_dict_section][conf_dict_name]
        ## No path was found
        else: 
            path_var = ""
            temp_str = "No directory path was found for {}.".format(name)
            self.add_textbox(**{
                        "text": temp_str,
                        "text_loc": [0.025,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
            ## Set value in report_conf
            self.report_conf["arguments"][report_conf_name] = path_var
            return
        
        ## Set value in report_conf
        self.report_conf["arguments"][report_conf_name] = path_var
        
        ## Check if directory exists
        if len(path_var) > 0:
            if not os.path.exists(path_var):
                temp_str = ["Path was not found for {}. Attempted path was: {}"
                            .format(name, path_var)]
                self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.025,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
                ## Because the path doesn't exist, we will have to reset the 
                ## argument in report_conf
                self.report_conf["arguments"][report_conf_name] = ""
            else:
                ## Path was found. Add this to report
                temp_str = "Path used for {} was: {}".format(name, path_var)
                self.add_textbox(**{
                        "text": temp_str,
                        "text_loc": [0.025,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                            }}
                        )
        else:
            temp_str = ["No path found for {}.".format(name)]
            self.add_textbox(**{
                        "text": temp_str[0],
                        "text_loc": [0.025,0.5],
                        "height": 1,
                        "edgecolor": "w",
                        "text_kw": 
                            {
                                "horizontalalignment": "left",
                                "fontsize": 12,
                                "wrap": True,
                                "color": "tab:red",
                            }}
                        )
                
        return 
        
    
    def get_acsf_calc(self):
        """
        Initializes and returns the acsf calculator to be parallelized over.
        
        """
        rsf = RSF(**self.report_conf["arguments"]["RSF_kw"])
        driver = Driver(rsf, **self.report_conf["arguments"]["driver_kw"])
        return driver
    
    
    ## Below are functions for getting principles components and plotting. 
    ## It's assumed from this point on that all functions will only be called
    ## by rank 0.
    
    def plot_pca(self):
        """
        Master plotting function that controls PCA calculation and plotting.
        
        """
        self.report_conf["Plot"] = {}
        name_list,matrix_list,struct_id_list,mean_list,std_list,target_list,pca_list = \
            self.calculate_pca() 
        
        ## For PCA plotting API
        exp_data = {}
        if len(self.report_conf["arguments"]["exp_path"]) > 0:
            s = read(self.report_conf["arguments"]["exp_path"])
            features = np.array([x.properties[self.acsf_key] for x in s.values()])
            exp_data = {"features": np.array(features),
                        "energy":  0}
            
        order = ["Raw", "AP1", "AP2", "Relaxed"]
        close_list = []
        
        for name in order:
            if name in name_list:
                idx = name_list.index(name)
            else:
                continue
            
            ## Need to normalize features using same mean and std as when the 
            ## PCA vectors were created
            mean = mean_list[idx]
            std = std_list[idx]
            
            if len(self.report_conf["arguments"]["exp_path"]) > 0:
                norm_features = (np.array(exp_data["features"]) - mean) / std
                temp_exp_data = {"features": norm_features.tolist(),
                                 "energy":  0,
                                 "norm": False}
            else:
                norm_features = []
                temp_exp_data = {"features": [],
                                 "energy": 0,
                                 "norm": False}
                
            
            ## Also need to normalize matrix=
            matrix = np.array(matrix_list[idx])
            norm_matrix = (matrix - mean) / std
            
            pca_kw = \
                {
                    "matrix": norm_matrix.tolist(),
                    "targets": target_list[idx],
                    "pca_components": pca_list[idx],
                    "exp_data": temp_exp_data,
                    "norm": False,
                }
            call = ["self.call_pca('{}', {}, ax)"
                    .format(name, pca_kw)]
            self.centered_plot(call[0], 
                              height=5,
                              ncols_total=5,
                              ncols_plot=3,
                              label="{} Pool".format(name))
        
        ### TO BE IMPLEMENTED
            close_list.append(self.close_structs(norm_matrix, 
                                                 norm_features, 
                                                 pca_list[idx],
                                                 struct_id_list[idx]))
        
        if len(self.report_conf["arguments"]["exp_path"]) > 0:
            self.add_header("Top {} Closest to Experimental"
                    .format(self.report_conf["arguments"]["close_n_structs"]))
            
            for name in order:
                if name in name_list:
                    idx = name_list.index(name)
                else:
                    continue
                
                name = name_list[idx]
                close = close_list[idx]
                temp_str = ["{}: {}".format(name, close)]
                temp_textbox_kw = self.info_textbox_kw.copy()
                temp_textbox_kw["height"] = 4
                self.add_textbox(temp_str[0], **temp_textbox_kw)
        
    
    def call_pca(self, name, pca_kw, ax):
        self.report_conf["Plot"][name] = pca_exp(**pca_kw, ax=ax)


    def close_structs(self, matrix, exp, pc, id_list):
        """
        Finds the closests structures to the experimental in the projection. 
        
        """
        if len(exp) == 0:
            return []
        matrix = np.array(matrix)
        pc = np.array(pc)
        exp = np.array(exp)
        proj = np.dot(matrix, pc.T)
        proj_exp = np.dot(exp, pc.T)
        dist = cdist(proj_exp, proj)
        dist = dist.reshape(-1)
        order = np.argsort(dist)
        order = order[:self.report_conf["arguments"]["close_n_structs"]]
        return [id_list[x] for x in order]

    
    def calculate_pca(self):
        """
        Will calculate the appropriate PCA vectors depending on the mode of 
        operation that the user has defined. The PCA vectors will be saved in 
        self.report_conf for each step of the downsmapling for reproducibility.
        Will call the respective function for the PCA mode. 
        
        Returning from this function will be a list of names, a list of 
        matrices of the ACSF values, a list of targets for each system, 
        and a list of principle components for projection.
        
        """
        ## Assume all principle components need to be recalculated. This 
        ## allows easy mode switching. Additionally, PCA calculations are 
        ## fast enough that this is not a huge issue. 
        
        ## Now check one by one if the section should be calculated
        ## Based on whether there's information to calculate the section
        name_list = []
        matrix_list = []
        struct_id_list = []
        mean_list = []
        std_list = []
        target_list = []
        pca_list = []
        s_raw = {}
        
        if len(self.report_conf["arguments"]["rsf_path_raw"]) > 0:
            path = self.report_conf["arguments"]["rsf_path_raw"]
            s_raw = read(path)
            name_list.append("Raw")
            temp_matrix, temp_ids = self.get_acsf_matrix(s_raw)
            matrix_list.append(temp_matrix)
            struct_id_list.append(temp_ids)
            target_list.append([])
        
        if len(self.report_conf["AP1"]["struct_id"]) > 0:
            s_ap1 = {}
            for struct_id in self.report_conf["AP1"]["struct_id"]:
                if s_raw.get(struct_id):
                    s_ap1[struct_id] = s_raw[struct_id]
            
            name_list.append("AP1")
            temp_matrix, temp_ids = self.get_acsf_matrix(s_ap1)
            matrix_list.append(temp_matrix)
            struct_id_list.append(temp_ids)
            target_list.append(self.report_conf["AP1"]["energy"])
        
        if len(self.report_conf["AP2"]["struct_id"]) > 0:
            s_ap2 = {}
            for struct_id in self.report_conf["AP2"]["struct_id"]:
                if s_raw.get(struct_id):
                    s_ap2[struct_id] = s_raw[struct_id]
            
            name_list.append("AP2")
            temp_matrix, temp_ids = self.get_acsf_matrix(s_ap2)
            matrix_list.append(temp_matrix)
            struct_id_list.append(temp_ids)
            target_list.append(self.report_conf["AP2"]["energy"])
            
        if len(self.report_conf["arguments"]["rsf_path_relaxed"]) > 0:
            path = self.report_conf["arguments"]["rsf_path_relaxed"]
            s_relaxed = read(path)
            name_list.append("Relaxed")
            temp_matrix, temp_ids = self.get_acsf_matrix(s_relaxed)
            matrix_list.append(temp_matrix)
            struct_id_list.append(temp_ids)
            results = get(s_relaxed, "prop", ["energy"])
            keep_idx = np.where(results["energy"].values != 0)[0]
            keep_results = results["energy"].values[keep_idx]
            target_list.append(keep_results.tolist())
        
        pca_list,mean_list,std_list = eval(
                        "self.pca_mode_{}(name_list, matrix_list)"
                        .format(self.report_conf["arguments"]["mode"]))
            
        
        return (name_list,matrix_list,struct_id_list,
                mean_list,std_list,target_list,pca_list)

    
    def pca_mode_Individual(self, name_list, matrix_list):
        """
        PCA calculations for individual projection. 
        
        """
        pca_list = []
        mean_list = []
        std_list = []
        for idx,name in enumerate(name_list):
            matrix = matrix_list[idx]
            
            ## Now normalize by columns such that one feature is not weighted 
            ## differently than another
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0)
            matrix = (matrix - mean) / std
            
            pca_list.append(self.get_pca(matrix))
            mean_list.append(mean)
            std_list.append(std)
        
        return pca_list, mean_list, std_list
            
    
    
    def pca_mode_AP(self, name_list, matrix_list):
        """
        PCA vectors will be calculated using the raw structures. These PCA
        vectors will be kept for the downsampling process but not for 
        the relaxed structures. This is because the unrelaxed structures
        may exhibit packing patterns that are not present in relaxed 
        structures. 
        
        """
        pca_list = []
        mean_list = []
        std_list = []
        
        ## Check if name list is compatible with pca mode
        if "Raw" in name_list:
            idx = name_list.index("Raw")
        elif "AP1" in name_list:
            idx = name_list.index("AP1")
        elif "AP2" in name_list:
            idx = name_list.index("AP2")
        else:
            raise Exception("Data not compatible with AP mode")
        
        ## Base PCA calculations using idx found above
        matrix = matrix_list[idx]
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
        pc_AP = self.get_pca(matrix)
        
        ## Now add mean, std, and pc for all but relaxed
        for idx,name in enumerate(name_list):
            if name == "Relaxed":
                ## Calculate for Relaxed
                matrix_relaxed = matrix_list[idx]
                mean_relaxed = np.mean(matrix_relaxed, axis=0)
                std_relaxed = np.std(matrix_relaxed, axis=0)
                matrix_relaxed = (matrix_relaxed - mean_relaxed) / std_relaxed
                pc_relaxed = self.get_pca(matrix_relaxed)
                mean_list.append(mean_relaxed)
                std_list.append(std_relaxed)
                pca_list.append(pc_relaxed)
            else:
                ## Otherwise add values already computed
                mean_list.append(mean)
                std_list.append(std)
                pca_list.append(pc_AP)
                
        return pca_list, mean_list, std_list
    
    
    def pca_mode_Raw(self, name_list, matrix_list):
        """
        Same as AP mode, but use the vectors found for unrelaxed structures 
        for the relaxed pool as well. 
        
        """
        pca_list = []
        mean_list = []
        std_list = []
        
        ## Check if name list is compatible with pca mode
        if "Raw" in name_list:
            idx = name_list.index("Raw")
        elif "AP1" in name_list:
            idx = name_list.index("AP1")
        elif "AP2" in name_list:
            idx = name_list.index("AP2")
        else:
            raise Exception("Data not compatible with Raw mode")
        
        ## Base PCA calculations using idx found above
        matrix = matrix_list[idx]
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
        pc_AP = self.get_pca(matrix)
        
        ## Now add mean, std, and pc for all
        for idx,name in enumerate(name_list):
            mean_list.append(mean)
            std_list.append(std)
            pca_list.append(pc_AP)
        
        return pca_list, mean_list, std_list
    
    
    def pca_mode_Relaxed(self, name_list, matrix_list):
        pca_list = []
        mean_list = []
        std_list = []
        
        ## Check if name list is compatible with pca mode
        if "Relaxed" in name_list:
            idx = name_list.index("Relaxed")
        else:
            raise Exception("Data not compatible with Relaxed mode")
        
        ## Base PCA calculations using idx found above
        matrix = matrix_list[idx]
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
        pc_relaxed = self.get_pca(matrix)
        
        ## Now add mean, std, and pc for all
        for idx,name in enumerate(name_list):
            mean_list.append(mean)
            std_list.append(std)
            pca_list.append(pc_relaxed)
        
        return pca_list, mean_list, std_list
    
    
    def get_acsf_matrix(self, struct_dict):
        """
        Gets the acsf matrix for a Structure dictionary and also returns the 
        structure ids that make up the matrix. 
        
        """
        results = get(struct_dict, "prop", [self.acsf_key, "energy"])
        keep_idx = np.where(results["energy"].values != 0)[0]
        keep_results = results[self.acsf_key].values[keep_idx]
        keep_results = np.vstack([np.array(x) for x in keep_results])
        struct_id = results.index[keep_idx]
        struct_id = struct_id.tolist()
        return keep_results,struct_id
    
    
    def get_pca(self, matrix):
        """
        Returns the principle components of the matrix. 
        
        """
        pca_obj = PCA(n_components=2)
        _ = pca_obj.fit_transform(matrix)
        return pca_obj.components_.tolist()


def correct_energy(energy_list, nmpc=4, global_min=None):
    '''
    Purpose:
        Corrects all energies values in the energy list relative to the global
          min and converts eV to kJ/mol/molecule.
    '''
    if global_min == None:
        global_min = min(energy_list)
    corrected_list = []
    for energy_value in energy_list:
        corrected_energy = energy_value - global_min
        corrected_energy = corrected_energy/(0.010364*float(nmpc))
        corrected_list.append(corrected_energy)
    return corrected_list


"""
Lessons for the future:
    - Manage all paths at the beginning and then store the final paths in the 
      report_conf["arguments"]. From then on, only reference valuess from 
      report_conf. 

"""
