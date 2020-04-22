# -*- coding: utf-8 -*-

import os,sys,json,copy
from configparser import ConfigParser

import numpy as np

import ibslib
from ibslib.io.aims_extractor import AimsExtractor
from ibslib.io.write import check_dir
from ibslib.io import read,write
from ibslib.molecules.volume_estimator import MoleculeVolumeEstimator
from ibslib.analysis import get
from ibslib.plot.genarris import labels_and_ticks, plot_volume_hist, plot_spg_hist
from ibslib.report.tag import tag_aspect,plot_tag,plot_time
from ibslib.plot.basic import *
from ibslib.plot.genarris import *
from ibslib.plot.gator import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Report_():
    """
    Report containing primitive functions to be inherited by other report 
    classes.
    
    Arguments
    ---------
    report_name: str
        File name the report file will be saved as inside the report folder. 
    report_folder: str
        Folder for performing report operations and saving the report file. 
    use_remake: bool
        Controls whether the report should be simply remake with the 
        section_calls.json file if it exists in the report_folder. If True, the
        report will first look for the remake file. If it exists, it will
        only use those attributes in this file.
    report_width: float
        Value, in inches, for how wide the report paper should be.
    
    Need to refactor section calls into the follow formatted arguments:
        report_call(("Name of section to add to plot",
                    "Import the Plot Function (Or any other function necessary)",
                    "Provide the name of the function for plotting",
                    "Dictionary of Arguments to the function",
                    overwrite_arguments=True))
        1) Everything has to be passed as a string. The reason for this is 
        so that all of the arguments can be written in the remake.json file. 
        For example, it may be easier to just pass the actual function in 
        memory with the actual arguments in memory. However, there's no 
        way this can be stored so easily. It may be more work for the developer,
        but the benefit of doing it this way is large. 
        
        2) I guess the dictionary of arguments does not necessarily have to be 
        a string. However, everything in the string should be checked to 
        ensure that it can be written into the remake.json. There should be 
        some automated ways to check convert everything to a json serializable 
        format. Alternatively, probably better to use smoe pickle format or
        a hdf5. Something that includes some compression. Maybe this will just
        be an option in the future to specify the file type of the remake json. 
        However, sometimes more user options is not always better. 
        
        3) Arguments to the report_call should be replaced if the 
        overwrite_arguments is True. This is useful for compatability to the 
        graphing API, which seems to work nicely. 
        
        4) All sections in the plot will be saved by a number appended to the 
        front of the name. This will guarantee that the names are sorted 
        in the correct order in the remake.json and will also guarantee that 
        all sections will have different names, even if it the name provided
        is the same. 
    
    """      
    def __init__(self, 
                report_name="report.pdf", 
                report_folder="report", 
                use_remake=False, 
                report_width=8.5):
        
        # Save path the report file will be saved at for future use
        self.report_path = os.path.join(report_folder, report_name)
        self.report_json_path = os.path.join(report_folder,
                                             "settings.json")
        
        # Initialize information for the actual figure of the report
        self.figure_kw = \
            {
              # Use standard paper width as default
              "figsize": [report_width,0],
              "constrained_layout": True,
            }
        self.gridspec_kw = \
            {
              "nrows": 0,
              "ncols": 1,
            }
        
        self.section_calls = []
        self.returned_section_calls = []
        
        #### Some storage for later calculations
        self.report_folder = report_folder
        check_dir(self.report_folder)
        self.report_name = report_name
        self.report_path = os.path.join(report_folder, report_name)
        self.remake_path = os.path.join(report_folder,"remake.json")
        self.cwd = os.path.abspath(os.getcwd())
        self.report_conf = {}
        self.remake = use_remake
        
        
    def report(self, 
               figname="", 
               write_remake=True, 
               write_json=True,
               timestamp=True,
               tag=True, 
               savefig_kw={"dpi": 600}):
        """
        Creates report folder with report, the settings file, and a script to
        remake the report from the settings file. 
        
        Idea here, is instead of saving all of the information similar to the 
        Genarris report, the commands ran in _generate_report can be stored
        simply and then evaluated again to create the same report. However, 
        it's still necessary to save the arguments returned from each 
        plotting script...
        
        """
        ### Check if remake file exists in report folder
        if self.remake:
            remake_path = os.path.join(self.report_folder, 
                                       "remake.json")
            if os.path.exists(remake_path):
                self.load_remake()
                
        ## Add ibslib tag. User can easily turn this off if desired. 
        if timestamp == True:
            self.add_time()
        if tag == True:
            self.add_tag()
        
                    
        self._generate_report(figname=figname, savefig_kw=savefig_kw)
        
        if write_json:
            print("Writing Json")
            self.write_json()
        
        if write_remake: 
            self.write_remake()
        
        
        
    def _generate_report(self, figname="", savefig_kw={"dpi": 600}):
        """
        Generates the actual report by using the information that has been
        created by each of the section methods.
        
        """
        fig = plt.figure(**self.figure_kw)
        gs = gridspec.GridSpec(**self.gridspec_kw, figure=fig)
        for command_list in self.section_calls:
            for command in command_list:
                try: exec(command)
                except Exception as e:
                    raise Exception("Report generation failed with "+
                            "error message {} \n".format(e) +
                            "for command {}".format(command))
        
        if len(figname) > 0:
            fig.savefig(figname, **savefig_kw)
        else:
            fig.savefig(self.report_path, **savefig_kw)
        
        
    def write_json(self, json_path=""):
        """
        Output a json file that has all agruments to regenerate the exact
        same report, perhaps without reference to any of the original 
        files. 
        
        """
        temp_dict = {}
        temp_dict["Report Configuration"] = self.report_conf
        
        if len(json_path) == 0:
            json_path = self.report_json_path
        
        with open(json_path,"w") as f:
            f.write(json.dumps(temp_dict, indent=4))
            
    
    def load_remake(self):
        """
        Loads the remake file.
        
        """
        remake_path = os.path.join(self.report_folder, 
                                       "remake.json")
        temp_dict = {}
        with open(remake_path) as f:
            temp_dict = json.load(f)
        
        self.section_calls =  temp_dict["section_calls"] 
        self.figure_kw = temp_dict["figure_kw"] 
        self.gridspec_kw = temp_dict["gridspec_kw"]
        self.report_path = temp_dict["report_path"]
        self.cwd = temp_dict["cwd"]
    
    
    def write_remake(self):
        """
        Writes the remake file. Only call this at the very end of operation 
        because it will change internal values.
        
        """
        remake_path = os.path.join(self.report_folder, 
                                       "remake.json")
        
        ## Only copy necessary values from self.__dict__ because otherwise
        ## it's not guaranteed to be json serializable
        temp_dict = {}
        temp_dict["section_calls"] = self.__dict__["section_calls"]
        temp_dict["figure_kw"] = self.__dict__["figure_kw"]
        temp_dict["gridspec_kw"] = self.__dict__["gridspec_kw"]
        temp_dict["report_name"] = self.report_name
        
        temp_dict["report_path"] = "./{}".format(self.report_name)
        temp_dict["cwd"] = "./"
        
        with open(remake_path, "w") as f:
            f.write(json.dumps(temp_dict))
            
    
    ########## Functions below are to be added to a more general
    #### Report class that will later be inherited
    def add_header(self, 
                   text, 
                   text_kw =
                       {
                         "horizontalalignment": "left",
                         "fontweight": "bold",
                         "fontsize": 16,
                         "color": "k",
                       },
                    edgecolor = "k",
                   ):
        """
        Primitive function. Adds a header to the report document.
        
        """
        # Adds row
        self.gridspec_kw["nrows"] += 1
        spec_idx = self.gridspec_kw["nrows"] - 1
        # Add space to figure for header
        self.figure_kw["figsize"][1] += 1
        temp_ax_command = \
            [
              # Use -1 because gs is zero-indexed
              "ax_{} = fig.add_subplot(gs[{}])".format(spec_idx,spec_idx),
              "ax_{}.set_xticks([])".format(spec_idx),
              "ax_{}.set_yticks([])".format(spec_idx),
              "ax_{}.text(0.05,0.5,\"{}\",**{})"
              .format(spec_idx,text, text_kw),
              "for spine in ax_{}.spines.values():  spine.set_edgecolor('{}')"
              .format(spec_idx, edgecolor)
            ]

        self.section_calls.append(temp_ax_command)
    
    
    def add_textbox(self, 
                    text, 
                    text_loc=[0.025,0.5],
                    height=2,
                    edgecolor="w",
                    text_kw =
                        {
                          "horizontalalignment": "left",
                          "fontsize": 12,
                          "wrap": True,
                        },
                    ):
        """
        Primitive function. Adds a textbox to the report.
        
        """
        self.gridspec_kw["nrows"] += height
        self.figure_kw["figsize"][1] += height
        
        # Add subplot across the entire row range of the height variable
        start_spec_idx = self.gridspec_kw["nrows"] - height
        end_spec_idx = self.gridspec_kw["nrows"]
        temp_ax_command = \
            [
              # Use -1 because gs is zero-indexed
              "ax_{} = fig.add_subplot(gs[{}:{},:])".format(start_spec_idx,
                                                          start_spec_idx,
                                                          end_spec_idx),
              "ax_{}.set_xticks([])".format(start_spec_idx),
              "ax_{}.set_yticks([])".format(start_spec_idx),
              "ax_{}.text({},{},\"{}\",clip_on=True,**{})".format(start_spec_idx,
                                                        text_loc[0],
                                                        text_loc[1],
                                                        text, 
                                                        text_kw),
              "for spine in ax_{}.spines.values():  spine.set_edgecolor('{}')"
              .format(start_spec_idx, edgecolor)
            ]
        
        self.section_calls.append(temp_ax_command)
    
    
    def add_row_plots(self, command_list, height=5):
        """
        Primitive function. Adds plots to a single row in the report.
        Command list has to use ax for plotting functionality.
        
        Argumnets
        ---------
        arguments: list
            List of a single command being used for the creation of each plot. 
            Each command should call a single function that creates the plot
            using all the required arguments for the plot.
        height: int
            Number of rows to use to add subplots. 
        """
        num_subplots = len(command_list)
        self.gridspec_kw["nrows"] += height
        self.figure_kw["figsize"][1] += height
        
        # Add subplot across the entire row range of the height variable
        start_spec_idx = self.gridspec_kw["nrows"] - height
        end_spec_idx = self.gridspec_kw["nrows"]
        temp_ax_command = \
            [
                # Need to split current subplot into the same number of 
                # subplots as graphs
                "inner_grid = gridspec.GridSpecFromSubplotSpec({}, {},"
                .format(1,num_subplots)
                +"subplot_spec=gs[{}:{},:], wspace=0.0, hspace=0.0)"
                .format(start_spec_idx,end_spec_idx)
            ]
        for i,command in enumerate(command_list):
            temp_command = "ax = fig.add_subplot(inner_grid[{}])".format(i)
            temp_ax_command.append(temp_command)
            temp_ax_command.append(command)
            
        self.section_calls.append(temp_ax_command)
    
    
    def centered_plot(self, 
                      plot_command, 
                      height=5, 
                      ncols_total=5,
                      ncols_plot=3,
                      label = "",
                      label_kw = 
                          {
                             "x": 0.5,
                             "y": 0.5,
                             "ha": "center",
                             "va": "center",
                             "fontweight": "bold",
                             "fontsize": 24,
                             "rotation": 90,
                             "color": "tab:red",
                          }
                      ):
        """
        Primitive function. Adds a plot centered in the report. 
        
        Arguments
        ---------
        plot_command:
            Command to create the plot.
        ncols_total: int
            Total number of columns to break the line into. 
        ncols_plot: int
            Number of columns the actual plot should span in the center of the
            image.
        
        """
        if ncols_plot > ncols_total:
            ncols_total = ncols_plot
        
        self.gridspec_kw["nrows"] += height
        self.figure_kw["figsize"][1] += height
        
        # Add subplot across the entire row range of the height variable
        start_spec_idx = self.gridspec_kw["nrows"] - height
        end_spec_idx = self.gridspec_kw["nrows"]
        num_subplots = ncols_total
        
        temp_ax_command = \
            [
                # Need to split current subplot into the same number of 
                # subplots as graphs
                "inner_grid = gridspec.GridSpecFromSubplotSpec({}, {},"
                .format(1,num_subplots)
                +"subplot_spec=gs[{}:{},:], wspace=0.0, hspace=0.0)"
                .format(start_spec_idx,end_spec_idx)
            ]
        
        ## If a label is to be used
        if len(label) > 0:
            temp_ax_command.append("ax_label = fig.add_subplot(inner_grid[:,0])")
            label_kw["s"] = label
            temp_ax_command.append("ax_label.text(**{})".format(label_kw))
            temp_ax_command.append("ax_label.set_frame_on(False)")
            temp_ax_command.append("ax_label.set_xticks([])")
            temp_ax_command.append("ax_label.set_yticks([])")
        
        ## Compute location of graph 
        offset = int((ncols_total - ncols_plot) / 2)
        remainder = ncols_total - ncols_plot - 2*offset
        center_start = offset
        center_end = num_subplots - offset - remainder
        temp_ax_command += ["ax = fig.add_subplot(inner_grid[:,{}:{}])"
                            .format(center_start, center_end)]
        temp_ax_command.append(plot_command)
        
        self.section_calls.append(temp_ax_command)


    def add_time(self):
        ## Check if time has already been added before
        if len(self.section_calls) > 2 and \
           self.section_calls[-1][-2] == "plot_time(ax)":
               return

        self.gridspec_kw["nrows"] += 1
        self.figure_kw["figsize"][1] += 1

        start_spec_idx = self.gridspec_kw["nrows"] - 1
        
        ax_command = ["ax = fig.add_subplot(gs[{}:-1])".format(start_spec_idx)]
        ax_command.append("ax.set_frame_on(False)")
        ax_command.append("ax.set_xticks([])")
        ax_command.append("ax.set_yticks([])")
        ax_command.append("plot_time(ax)")
        
        self.section_calls.append(ax_command)

        
    def add_tag(self):
        ## Check if tag has already been added before
        if len(self.section_calls) > 0 and \
           self.section_calls[-1][-1] == "plot_tag(ax)":
               return
           
        tag_width = self.figure_kw["figsize"][0]
        tag_height = tag_width*0.245
        
        self.gridspec_kw["nrows"] += 1
        self.figure_kw["figsize"][1] += 1

        start_spec_idx = self.gridspec_kw["nrows"] - 1
        
        ax_command = ["ax = fig.add_subplot(gs[{}:])".format(start_spec_idx)]
        ax_command.append("ax.set_frame_on(False)")
        ax_command.append("ax.set_xticks([])")
        ax_command.append("ax.set_yticks([])")
        ax_command.append("ax.set_adjustable('datalim')")
        ax_command.append("plot_tag(ax)")
        
            
        self.section_calls.append(ax_command)
    



class GenarrisReport(Report_):
    """
    Class for generating a report of the current Genarris calculation.
    The folder structure of the Genarris calculation is obtained from the 
    user configuration file that was used to perform the calculation. 
    
    Arguments
    ---------
    conf_path: str 
        Path to the location of the Genarris configuration file.
    json_path: str
        Path to the json file that describes that settings using to generate
        the report. It will be easier to make small adjustments to the 
        plots by providing the path to the json file that contains all 
        plot settings. Json file will be generated dynamically once,
        then users may change any small keyword argument for each section that 
        was generated. 
    
    To Do
    -----
    - Incorporate use of the json file loading. This will be done by adding 
      an argument list for each one of the sections with these arguments 
      passed in if they exist.
    - The structure of the json file will be as follows:
        1. ["Genarris Configuration"] The configuration of the Genarris 
           calculation that was identified by the program. 
        2. ["Report Configuration"] Containing the relevant arguments for each section 
           that was used to generate the report.
    - Need to make keys for properties modular as well probably.
    - Also need to perform checks on whether the desired option even exists
      in the configuration file. However, this would never be an issue if 
      defaults were implemented correctly. I guess I could implement defaults
      correctly here if I want. I would just have to collect them from the 
      source code. 
    - For now, the easiest way to get graph arguments is probably to just 
      do a run of the plot, thus having to create two plots for every one
      plot. This is the simplest way to do it. This also opens the easy 
      possibility of saving each graph individually which cannot be done 
      now. I guess this is okay.
    - Need to implement a check before every file is read form the file 
      system basically. I can also implement a standard error message type
      textbox as well that would help expedite this.
    
    """
    def __init__(self, 
                 conf_path="", 
                 json_path="", 
                 report_name="report.pdf",
                 report_folder="report", 
                 report_width=8.5):
        
        # Save path the report file will be saved at for future use
        self.report_folder = report_folder
        check_dir(self.report_folder)
        self.report_path = os.path.join(report_folder, report_name)
        self.report_json_path = os.path.join(report_folder,
                                             "settings.json")
        
        # Setup the configuration dictionary that describes the settings the 
        # Genarris calculation was conducted with.
        self.conf_dict = {}
        if len(conf_path) > 0:
            self.conf_dict = parse_conf(conf_path)
            self.report_conf = {}
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
        self.section_calls = []
        
        #### Some storage for later calculations
        self.cwd = os.path.abspath(os.getcwd())
        self.molecule_struct = None
        self.general = []
        self.special = []
    
    
    def parse_json(self, json_path):
        with open(json_path) as f:
            temp_dict = json.load(f)
        self.conf_dict = temp_dict["Genarris Configuration"]
        self.report_conf = temp_dict["Report Configuration"]
        
    
    def write_json(self, json_path=""):
        """
        Output a json file that has all agruments to regenerate the exact
        same report, perhaps without reference to any of the original 
        files. 
        
        """
        temp_dict = {}
        temp_dict["Genarris Configuration"] = self.conf_dict
        temp_dict["Report Configuration"] = self.report_conf
        
        if len(json_path) == 0:
            json_path = self.report_json_path
        
        with open(json_path,"w") as f:
            f.write(json.dumps(temp_dict, indent=4))
    
    
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
                    "from ibslib.report import GenarrisReport\n" +
                    "json_path = \"settings.json\"\n" +
                    "rp = GenarrisReport(json_path=json_path,report_folder=\"./\", "+
                    "report_width={})\n".format(self.figure_kw["figsize"][0])+
                    "rp.report()"
                ]
            text = text[0]
            f.write(text)
    
    
    def report(self, output_dir="report", figname=""):
        """
        Implementation is simple and modular. For each section in the 
        conf_dict, call the associated GenarrisReport.<section>. Each of these
        methods will define the appropraite behavior for the section. Thus, 
        when new sections are added, the developer only needs to add a new
        method to this class for that section. Each method should prepare a 
        section to add to the figure. 
        
        """
        if len(self.conf_dict) == 0:
            raise Exception("The configuration for the report is empty. " +
                "Set the configuration by providing a configuration file or "+
                "a report json file to the construction of the "+
                "GenarrisReport class.")
        
        for i,section in enumerate(self.conf_dict):
            # No arguments provided for section
            if self.report_conf.get(section) == None:
                # Each section pass back the arugments that were used 
                # to create report configuration
                self.report_conf[section] = eval("self.{}()".format(section))
            else:
                self.report_conf[section] = eval("self.{}(**{})"
                                .format(section, self.report_conf[section]))
        
        self.add_tag()
        self._generate_report(figname=self.report_path)
        self.write_json()
        self.write_make_report()
    
    
    def _copy_arguments(self, arguments):
        """
        Helper function to use arugments=locals() and then remove all entries
        that should not actually be retained. 
        
        """
        arguments_copy = {}
        for key,value in arguments.items():
            if key == "self":
                continue
            elif key == "arguments":
                continue
            else:
                arguments_copy[key] = value
        return arguments_copy
    
    ######### Genarris specific sections
    def Genarris_master(
            self, 
            textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw": 
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                        }
                }):
        arguments = self._copy_arguments(locals())
        
        print("Genarris Master")
        
        self.add_header("Genarris Master")
        procedures = self.conf_dict["Genarris_master"]["procedures"]
        procedures = eval(procedures)
        procedures = ", ".join(procedures)
        text = "Procedures: \\n \\n" + procedures
        ## Set text as procedures if text is not already passed in
        if len(arguments["textbox_kw"]["text"]) == 0:
            arguments["textbox_kw"]["text"] = text
        self.add_textbox(**arguments["textbox_kw"])
        
        return arguments
        
    
    def relax_single_molecule(self, 
            error_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
            report_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.35],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 16,
                             "wrap": True,
                        }
                },):
        """
        1. Check if molecule path exists
        2. Check if aims calculation directory exists
        3. Check if aims calculation is done
        4. If done, print energy value and time it took to calculate. If not
        done, print in progress message.
        
        """
        arguments = self._copy_arguments(locals())
        
        print("Relax Single Molecule")
        
        self.add_header("Relax Single Molecule")
        
        ###### Check if generating from pre-made report
        if len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"])
            return arguments
        
        single_molecule_path = self.conf_dict["relax_single_molecule"]["molecule_path"]
        if not os.path.exists(single_molecule_path):
            text = ["Path to the molecule file {} was not found. "
                    .format(single_molecule_path)+
                    "Please check "+
                    "the molecule path is correct before proceeding."]
            text = text[0]
            
            if len(arguments["error_textbox_kw"]["text"]) == 0:
                arguments["error_textbox_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_kw"])
            return arguments
        
        
        #### Need to save structure for estimate_unit_cell_volume
        ### For now, just using single_molecule_path
        self.molecule_struct = read(single_molecule_path)
        
        #### Need to add another layer because that's just the way single 
        ## molecule calculation folder is set up for some reason. I don't
        ## approve.
        aims_dir = self.conf_dict["relax_single_molecule"]["aims_output_dir"]
        temp = read(single_molecule_path)
        aims_dir = os.path.join(aims_dir, temp.struct_id)
        if not os.path.exists(aims_dir):
            text = ["Path to FHI-aims calculation {} was not found."
                    .format(aims_dir) +
                    "Either the "+
                    "calculation has not started yet or the path provided "+
                    "is incorrect."]
            if len(arguments["error_textbox_kw"]["text"]) == 0:
                arguments["error_textbox_kw"]["text"] = text
                
            self.add_textbox(**arguments["error_textbox_kw"])
            return arguments
        
        #### Calculation has started. Check the calculation progress.
        ae = AimsExtractor(aims_dir, 
                           aims_property=["energy","time"],
                           energy_name="Energy")
        
        result = ae.find_aims_file(aims_dir)
        if result == '1':
            text = ["FHI-aims calculation directory exists, but no "+
                    "aims.out file was found. The calculation has likely "+
                    "not started yet."]
            text = text[0]
            text_kw = \
                    {
                      "horizontalalignment": "left",
                      "fontsize": 12,
                      "wrap": True,
                      "color": "tab:red",
                    }
            self.add_textbox(text, text_kw=text_kw)
            return
        
        
        aims_path = os.path.join(aims_dir,"aims.out")
        results = ae.extract_from_output(aims_path)
        
        text = ["Single molecule file has finished calculating.\\n"+
                "\\n" +
                "Total molecule energy: {} eV\\n".format(results["Energy"])+
                "\\n" +
                "Total calculation time: {} seconds".format(results["Total Calculation Time"])]
        text = text[0]
        
        if len(arguments["report_textbox_kw"]["text"]) == 0:
            arguments["report_textbox_kw"]["text"] = text
        
        self.add_textbox(**arguments["report_textbox_kw"])
        
        return arguments
              

    def estimate_unit_cell_volume(
            self,
            error_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.35],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
            report_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.35],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 16,
                             "wrap": True,
                        }
                }):
        """
        Need to recalculate because it was never saved by Genarris anywhere, 
        only print to the log file.
        
        """
        arguments = self._copy_arguments(locals())
        
        print("Estimate Unit Cell Volume")
        
        self.add_header("Volume Estimate")
        
        ###### Check if generating from pre-made report
        if len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"])
            return arguments
        elif len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"]["text"])
            return arguments
        
        if self.molecule_struct == None:
            text = ["Set GenarrisReport.molecule_struct before " +
                    "calling GenarrisReport.estimate_unit_cell_volume(). "+
                    "GenarrisReport.molecule_struct is set automatically "+
                    "after GenarrisReport completes the report for "+
                    "Relax Single Molecule. " +
                    "If the report for Relax Single Molecule hasn't been "+
                    "generated yet, this is the default behavior."]
            text = text[0]
            
            if len(arguments["error_textbox_kw"]["text"]) == 0:
                arguments["error_textbox_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_kw"])
            return arguments
        
        mve = MoleculeVolumeEstimator()
        mve = mve.calc_struct(self.molecule_struct)
        Z = int(self.conf_dict["estimate_unit_cell_volume"]["z"])
        self.molecule_struct.properties["predicted_molecule_volume"] = mve
        self.molecule_struct.properties["predicted_ucv"] = mve*Z
        # Hard code stdev for this module
        std = 3.0*0.025*mve
        text = ["Predicted molecule solid form volume: {:.2f} $\AA^3$\\n".format(mve) +
                "Predicted unit cell volume for Z={}: {:.2f} $\AA^3$\\n".format(Z,mve*Z) +
                "Standard Deviation of volume: {:.2f} $\AA^3$".format(std)]
        text = text[0]
        
        if len(arguments["report_textbox_kw"]["text"]) == 0:
            arguments["report_textbox_kw"]["text"] = text
        
        self.add_textbox(**arguments["report_textbox_kw"])
        
        return arguments
        
    
    
    def pygenarris_structure_generation(
            self, 
            tol=0.1,
            exp_volume = -1, 
            property_names = 
                {
                    "Space Group": "spg",
                    "Unit Cell Volume": "unit_cell_volume"
                },
            error_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
            report_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 1,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 14,
                             "wrap": True,
                        }
                },
             plot_arguments = 
                 {
                     "height": 3,
                     "plot_volume_hist": {},
                     "plot_spg_hist": {},
                 }
            ):
        """
        1. Checks if the folder containing raw pool exists yet
        2. Checks number of structures in the raw pool is equal to the 
           target number. 
        3. Plots the space group histogram of the structures
        4. Plots the volume histogram of the structures with the volume that
           was use to generate the pool.
        5. Create lattice parameter plots.
        6. Projection plots?
        
        """
        arguments = self._copy_arguments(locals())
        
        print("Structure Generation")
        self.add_header("Structure Generation")
        
        ###### Check if generating from pre-made report
        if len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"])
            self._row_plots_arguments("pygenarris_structure_generation",
                                      plot_arguments)
            return arguments
        elif len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"]["text"])
            return arguments
        
        ###### Want to generate space group and volume histograms that have 
        ## the same axis for structure generation, and AP downsampling.
        output_dir = self.conf_dict["pygenarris_structure_generation"]["output_dir"]
               
        if not os.path.exists(output_dir):
            text = ["Folder of generated structures {} ".format(output_dir)+
                    "was not found."]
            text = text[0]
            
            if len(arguments["error_textbox_kw"]["text"]) == 0:
                arguments["error_textbox_kw"]["text"] = text 
            
            self.add_textbox(**arguments["error_textbox_kw"])
            return arguments
        
        #### Check for number of structures generated
        struct_dict = read(output_dir)
        target = self.conf_dict["pygenarris_structure_generation"]["num_structures"]
        generated = len(struct_dict)
        
        text = ["Target number of structures: {}\\n".format(target)+
                "Number of structures generated: {}".format(generated)]
        text = text[0]
        
        if len(arguments["report_textbox_kw"]["text"]) == 0:
            arguments["report_textbox_kw"]["text"] =  text
        
        self.add_textbox(**arguments["report_textbox_kw"])
        
        ### Need to generate allow spacegroup file using Pygenarris
        
        #### Need to do this crappy stuff: Explicitly change directory and 
        ## explicitly create python function that is called in the terminal
        ## and then redirected to file. There isn't really a better way for 
        ## this to be done given that you cannot redirect the stdout of 
        ## the Swig wrapped function from inside Python itself.
        os.chdir(self.report_folder)
        write("geometry.in", self.molecule_struct, file_format="geo", overwrite=True)
        z = int(self.conf_dict["pygenarris_structure_generation"]["z"])
        python_file_text = \
            [
              "from pygenarris import num_compatible_spacegroups \n" +
              "num_compatible_spacegroups({},{})".format(z,tol)
            ]
        python_file_text = python_file_text[0]
        with open("compatible_spacegroups.py","w") as f:
            f.write(python_file_text)
        
        os.system("python compatible_spacegroups.py > compatible_spacegroups.txt")
        
        general = []
        special = []
        with open("compatible_spacegroups.txt") as f:
            for line in f:
                split_line = line.split()
                if 'spg' in split_line[0]:
                    spg = int(split_line[1])
                    position = split_line[-3]
                else:
                    continue
                # Add to general or special depending on site symmetry
                if position == "1":
                    # Check if already added
                    if spg in general:
                        continue
                    else:
                        general.append(spg)
                else:
                    if spg in special:
                        continue
                    else:
                        special.append(spg)
        
        self.general = general
        self.special = special
        
        os.chdir(self.cwd)
        
        
        #### Now use the general _row_plots to add plots 
        ## for space groups and volumes
        self._row_plots("pygenarris_structure_generation",
                        output_dir, 
                        height=plot_arguments["height"],
                        exp_volume = exp_volume,
                        property_names = property_names)
    
        return arguments
    
    
    def run_rdf_calc(self):
        arguments = self._copy_arguments(locals())
        
        print("RSF")
        
        self.add_header("RSF Calculation")
        return arguments
        
        dist_matrix_path = self.conf_dict["run_rdf_calc"]["dist_mat_fpath"]
        
        if not os.path.exists(dist_matrix_path):
            text = ["Distance matrix was not found at {}.".format(dist_matrix_path)+
                    "The distance calculation may not have completed yet. "+
                    "Double check that the given path is correct."]
            text = text[0]
            text_kw = \
                    {
                      "horizontalalignment": "left",
                      "fontsize": 14,
                      "wrap": True,
                      "color": "tab:red",
                    }
            self.add_textbox(text, text_kw=text_kw, text_loc=[0.025,0.5])
            return
        
        ### Load in distance matrix and make distance plot
        dist_matrix = np.memmap(dist_matrix_path, dtype="float32")
        sqrt_ = np.sqrt(dist_matrix.shape[0])
        num_struct = int(sqrt_)
        
        if num_struct < sqrt_:
            text = ["The square root of the distance matrix was {} ".format(sqrt_)+
                    "which is not a perfect integer. Please check that the "+
                    "distance matrix can be read correctly using the "+
                    "following command: np.memmap(dist_matrix_path, dtype=\"float32\")"]
            text = text[0]
            text_kw = \
                    {
                      "horizontalalignment": "left",
                      "fontsize": 14,
                      "wrap": True,
                      "color": "tab:red",
                    }
            self.add_textbox(text, text_kw=text_kw, text_loc=[0.025,0.5])
            return
        
        dist_matrix = np.array(dist_matrix.reshape(num_struct,num_struct))
        
        command_list = []
        command = ["plot_dist_mat({},".format(dist_matrix.tolist()) +
                   "ax=ax)"]
        command = command[0]
        command_list.append(command)
        
        self.add_row_plots(command_list, height=5)
        

    def run_rsf_calc(self):
        arguments = self._copy_arguments(locals())
        
        print("RSF")
        
        self.add_header("RSF Calculation")
        return arguments
        
        dist_matrix_path = self.conf_dict["run_rsf_calc"]["dist_mat_fpath"]
        
        if not os.path.exists(dist_matrix_path):
            text = ["Distance matrix was not found at {}.".format(dist_matrix_path)+
                    "The distance calculation may not have completed yet. "+
                    "Double check that the given path is correct."]
            text = text[0]
            text_kw = \
                    {
                      "horizontalalignment": "left",
                      "fontsize": 14,
                      "wrap": True,
                      "color": "tab:red",
                    }
            self.add_textbox(text, text_kw=text_kw, text_loc=[0.025,0.5])
            return
        
        ### Load in distance matrix and make distance plot
        dist_matrix = np.memmap(dist_matrix_path, dtype="float32")
        sqrt_ = np.sqrt(dist_matrix.shape[0])
        num_struct = int(sqrt_)
        
        if num_struct < sqrt_:
            text = ["The square root of the distance matrix was {} ".format(sqrt_)+
                    "which is not a perfect integer. Please check that the "+
                    "distance matrix can be read correctly using the "+
                    "following command: np.memmap(dist_matrix_path, dtype=\"float32\")"]
            text = text[0]
            text_kw = \
                    {
                      "horizontalalignment": "left",
                      "fontsize": 14,
                      "wrap": True,
                      "color": "tab:red",
                    }
            self.add_textbox(text, text_kw=text_kw, text_loc=[0.025,0.5])
            return
        
        dist_matrix = np.array(dist_matrix.reshape(num_struct,num_struct))
        
        command_list = []
        command = ["plot_dist_mat({},".format(dist_matrix.tolist()) +
                   "ax=ax)"]
        command = command[0]
        command_list.append(command)
        
        self.add_row_plots(command_list, height=5)
    
    
    def affinity_propagation_fixed_clusters(
            self,
            error_textbox_1_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
            report_textbox_1_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.25],
                    "height": 1,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 14,
                             "wrap": True,
                        }
                },
            error_textbox_2_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
            report_textbox_2_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.25],
                    "height": 1,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 14,
                             "wrap": True,
                        }
                },
            property_names = 
                {
                    "Space Group": "spg",
                    "Unit Cell Volume": "unit_cell_volume"
                },
            plot_arguments = 
                 {
                     "height": 3,
                     "plot_volume_hist_1": {},
                     "plot_spg_hist_1": {},
                     "plot_volume_hist_2": {},
                     "plot_spg_hist_2": {},
                 }):
        
        arguments = self._copy_arguments(locals())
        
        print("Downselection")
        
        self.add_header("AP Downsampling")
        
        ###### Check if generating from pre-made report
        if len(arguments["report_textbox_1_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_1_kw"])
            self._row_plots_arguments("affinity_propagation_fixed_clusters",
                                      plot_arguments,
                                      AP_step=1)
        elif len(arguments["error_textbox_1_kw"]["text"]) != 0:
            self.add_textbox(**arguments["error_textbox_1_kw"])
        
        if len(arguments["report_textbox_2_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_1_kw"])
            self._row_plots_arguments("affinity_propagation_fixed_clusters",
                                      plot_arguments,
                                      AP_step=2)
            return arguments
        elif len(arguments["error_textbox_2_kw"]["text"]) != 0:
            self.add_textbox(**arguments["error_textbox_2_kw"])
            return arguments
        
        num_clusters_1 = self.conf_dict["affinity_propagation_fixed_clusters"]["num_of_clusters"]
        num_clusters_2 = self.conf_dict["affinity_propagation_fixed_clusters"]["num_of_clusters_2"]
        tol = self.conf_dict["affinity_propagation_fixed_clusters"]["num_of_clusters_tolerance"]
        
        exemplars_output_dir_1 = self.conf_dict["affinity_propagation_fixed_clusters"]["exemplars_output_dir"]
        exemplars_output_dir_2 = self.conf_dict["affinity_propagation_fixed_clusters"]["exemplars_output_dir_2"]
        
        clustered_structs_1 = self.conf_dict["affinity_propagation_fixed_clusters"]["output_dir"]
        clustered_structs_2 = self.conf_dict["affinity_propagation_fixed_clusters"]["output_dir_2"]
                               
        if not os.path.exists(exemplars_output_dir_1):
            text = ["AP could not find first exemplars output directory {}. "
                    .format(exemplars_output_dir_1)]
            text = text[0]
            if len(arguments["error_textbox_1_kw"]["text"]) == 0:
                arguments["error_textbox_1_kw"]["text"] = text
            self.add_textbox(**arguments["error_textbox_1_kw"])
        else:
            s = read(exemplars_output_dir_1)
            text = ["AP Clustering step 1. \\n"+
                    "Target Structures: {}\\n".format(num_clusters_1)+
                    "Structures Found: {}".format(len(s))]
            
            text = text[0]
            if len(arguments["report_textbox_1_kw"]["text"]) == 0:
                arguments["report_textbox_1_kw"]["text"] = text
            self.add_textbox(**arguments["report_textbox_1_kw"])
            self._row_plots("affinity_propagation_fixed_clusters",
                            exemplars_output_dir_1,
                            height=plot_arguments["height"],
                            property_names=property_names,
                            AP_step=1)
        
        if not os.path.exists(exemplars_output_dir_2):
            text = ["AP could not find second exemplars output directory {}. "
                    .format(exemplars_output_dir_2)]
            text = text[0]
            if len(arguments["error_textbox_2_kw"]["text"]) == 0:
                arguments["error_textbox_2_kw"]["text"] = text
            self.add_textbox(**arguments["error_textbox_2_kw"])
        else:
            s = read(exemplars_output_dir_2)
            text = ["AP Clustering step 2. \\n"+
                    "Target Structures: {}\\n".format(num_clusters_2)+
                    "Structures Found: {}".format(len(s))]
            text = text[0]
            
            if len(arguments["report_textbox_2_kw"]["text"]) == 0:
                arguments["report_textbox_2_kw"]["text"] = text
            
            self.add_textbox(**arguments["report_textbox_2_kw"])
            
            self._row_plots("affinity_propagation_fixed_clusters",
                            exemplars_output_dir_2,
                            height=plot_arguments["height"],
                            property_names=property_names,
                            AP_step=2)
        
        return arguments
        

    def fhi_aims_energy_evaluation(
            self,
            error_textbox_1_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
             error_textbox_2_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                },
            report_textbox_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.25],
                    "height": 1,
                    "edgecolor": "w",
                    "text_kw":
                        {
                             "horizontalalignment": "left",
                             "fontsize": 14,
                             "wrap": True,
                        }
                },
            property_names = 
                {
                    "Energy": "energy"
                }
            ):
        arguments = self._copy_arguments(locals())
        
        print("SCF Energy")
        
        self.add_header("SCF Energy")
        
        ###### Check if generating from pre-made report
        if len(arguments["error_textbox_1_kw"]["text"]) != 0 or \
            len(arguments["error_textbox_2_kw"]["text"]) != 0:
            if len(arguments["error_textbox_1_kw"]["text"]) != 0:
                self.add_textbox(**arguments["error_textbox_1_kw"])
            if len(arguments["error_textbox_2_kw"]["text"]) != 0:
                self.add_textbox(**arguments["error_textbox_2_kw"])
            return arguments
        elif len(arguments["report_textbox_kw"]["text"]) != 0:
            self.add_textbox(**arguments["report_textbox_kw"])
            return arguments
        
        aims_calc_dir = self.conf_dict["fhi_aims_energy_evaluation"]["aims_output_dir"]
        struct_dir = self.conf_dict["fhi_aims_energy_evaluation"]["output_dir"]
        energy_name = self.conf_dict["fhi_aims_energy_evaluation"]["energy_name"]
        
        if not os.path.exists(aims_calc_dir):
            text = ["FHI-aims calculation directory {} ".format(aims_calc_dir) +
                    "for fhi_aims_energy_evaluation procedure was not found."]
            text = text[0]
            
            if len(arguments["error_textbox_1_kw"]["text"]) == 0:
                arguments["error_textbox_1_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_1_kw"])
        
        if not os.path.exists(struct_dir):
            text = ["Structure output directory {} ".format(aims_calc_dir) +
                    "for fhi_aims_energy_evaluation procedure was not found."]
            text = text[0]
            if len(arguments["error_textbox_2_kw"]["text"]) == 0:
                arguments["error_textbox_2_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_2_kw"])
        else:
            s = read(struct_dir)
            energy_name = property_names["Energy"]
            results = get(s, "prop", [energy_name])
            num_energy = len(np.where(results[energy_name].values != 0)[0])
            text = ["Found {} structures "
                    .format(len(s)) +
                    "with SCF energy values."]
            text = text[0]
            arguments["report_textbox_kw"]["text"] = text
            self.add_textbox(**arguments["report_textbox_kw"])
            
        return arguments
    
    
    def run_fhi_aims_batch(
            self,
            error_textbox_1_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                    
                },
             error_textbox_2_kw = 
                {
                    "text": "",
                    "text_loc": [0.025,0.5],
                    "height": 2,
                    "edgecolor": "w",
                    "text_kw":
                        {
                            "horizontalalignment": "left",
                            "fontsize": 12,
                            "wrap": True,
                            "color": "tab:red",
                        }
                },
            property_names = 
                {
                    "Space Group": "spg",
                    "Unit Cell Volume": "unit_cell_volume",
                    "Energy": "energy"
                },
            plot_arguments = 
                 {
                     "height": 3, 
                     "plot_volume_hist": {},
                     "plot_spg_hist": {},
                 }
            ):
        arguments = self._copy_arguments(locals())
        
        print("Relaxations")
        
        self.add_header("Relaxation")
        
         ###### Check if generating from pre-made report
        if len(arguments["error_textbox_1_kw"]["text"]) != 0 or \
            len(arguments["error_textbox_2_kw"]["text"]) != 0:
            if len(arguments["error_textbox_1_kw"]["text"]) != 0:
                self.add_textbox(**arguments["error_textbox_1_kw"])
            if len(arguments["error_textbox_2_kw"]["text"]) != 0:
                self.add_textbox(**arguments["error_textbox_2_kw"])
            return arguments
        elif len(plot_arguments["plot_volume_hist"]) != 0:
            self._row_plots_arguments("run_fhi_aims_batch",
                                      plot_arguments)
            return arguments
        
        aims_calc_dir = self.conf_dict["run_fhi_aims_batch"]["aims_output_dir"]
        struct_dir = self.conf_dict["run_fhi_aims_batch"]["output_dir"]
        energy_name = self.conf_dict["run_fhi_aims_batch"]["energy_name"]
        
        if not os.path.exists(aims_calc_dir):
            text = ["FHI-aims calculation directory {} ".format(aims_calc_dir) +
                    "for run_fhi_aims_batch procedure was not found."]
            text = text[0]
            if len(arguments["error_textbox_1_kw"]["text"]) == 0:
                arguments["error_textbox_1_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_1_kw"])

        
        if not os.path.exists(struct_dir):
            text = ["Structure output directory {} ".format(struct_dir) +
                    "for run_fhi_aims_batch procedure was not found."]
            text = text[0]
            if len(arguments["error_textbox_2_kw"]["text"]) == 0:
                arguments["error_textbox_2_kw"]["text"] = text
            
            self.add_textbox(**arguments["error_textbox_2_kw"])
            
        else:
            ### If everything was found then we can create the row of plots
            self._row_plots("run_fhi_aims_batch", struct_dir,
                            height=plot_arguments["height"],
                            property_names=property_names)
        
        return arguments
        
    
    def _row_plots(self, 
            section_name,
            struct_path, 
            exp_volume = -1,
            height=3,
            property_names = 
                {
                    "Space Group": "spg",
                    "Unit Cell Volume": "unit_cell_volume"
                },
            AP_step=1):
        """
        Defines behavior for the basic row of plots that are created for the 
        report of the Genarris calculations. 
        """
        struct_dict = read(struct_path)
        
        # Get space group, unit cell volume, and unit cell vector parameters
        spg_name = property_names["Space Group"]
        ucv_name = property_names["Unit Cell Volume"]
        results = get(struct_dict, "prop", [spg_name,ucv_name])
        
        spg_list =  results[spg_name].values.tolist()
        volume_list = results[ucv_name].values.tolist()
        
        ## Get predicted volume for later reference
        try: 
            pred_vol = self.molecule_struct.properties["predicted_ucv"]
        except:
            pred_vol = -1
        
        #### Now that the compatible space group information has been 
        ## collected, we may add the correct plots
        command_list = []
        
        ### Now add plot and save its arguments return from the function
        command = [
                   "self.report_conf[\"{}\"][\"plot_arguments\"][\"plot_spg_hist\"] = ".format(section_name) +
                   "plot_spg_hist({},".format(spg_list) +
                   "general_spg_values={},".format(self.general) +
                   "special_spg_values={},".format(self.special) +
                   "ax=ax)"]
        command = command[0]
        command_list.append(command)
        
        command = [
                   "self.report_conf[\"{}\"][\"plot_arguments\"][\"plot_volume_hist\"] = ".format(section_name) +
                   "plot_volume_hist({},".format(volume_list) +
                   "pred_volume={},"
                   .format(pred_vol) +
                   "ax=ax)\n"]
        command = command[0]
        command_list.append(command)
        
        #### Use different commands if section is for AP because the API for AP 
        ## wasn't implemented well in Genarris. So, now we have to do 
        ## special commands just for AP. The AP steps should really be 
        ## separated with a clean API in the Genarris source code.
        if section_name == "affinity_propagation_fixed_clusters":
            command_list = []
            command = [
                   "self.report_conf[\"{}\"][\"plot_arguments\"][\"plot_spg_hist_{}\"] = ".format(section_name,AP_step) +
                   "plot_spg_hist({},".format(spg_list) +
                   "general_spg_values={},".format(self.general) +
                   "special_spg_values={},".format(self.special) +
                   "ax=ax)"]
            command = command[0]
            command_list.append(command)
            
            command = [
                       "self.report_conf[\"{}\"][\"plot_arguments\"][\"plot_volume_hist_{}\"] = ".format(section_name,AP_step) +
                       "plot_volume_hist({},".format(volume_list) +
                       "pred_volume={},"
                       .format(pred_vol) +
                       "ax=ax)\n"]
            command = command[0]
            command_list.append(command)
        
        self.add_row_plots(command_list, height=height)
    
    
    def _row_plots_arguments(self,section_name,plot_arguments,
                             AP_step=1):
        """
        Making the Genarris report row plots simply from plot_arguments.
        
        """
        command_list = []
        if section_name != "affinity_propagation_fixed_clusters":
            command_list.append(   
                "plot_spg_hist(**{},ax=ax)".format(plot_arguments["plot_spg_hist"]))
            command_list.append(
                "plot_volume_hist(**{},ax=ax)".format(plot_arguments["plot_volume_hist"]))
        else:
            command_list.append(             
                "plot_spg_hist(**{}, ax=ax)".format(plot_arguments["plot_spg_hist_{}".format(AP_step)]))
            command_list.append(
                "plot_volume_hist(**{}, ax=ax)".format(plot_arguments["plot_volume_hist_{}".format(AP_step)]))
            
        self.add_row_plots(command_list, height=plot_arguments["height"])


def parse_conf(conf_path):
    if not os.path.exists(conf_path):
        raise Exception("Configuration file {} was not found.".format(conf_path))
    config = ConfigParser()
    config.read(conf_path)
    sections = config.sections()
    conf_dict = {}
    for section in sections:
        conf_dict[section] = {}
        for option in config[section]:
            conf_dict[section][option] = config[section][option]
            
    return conf_dict
                


if __name__ == "__main__":
    pass
    
