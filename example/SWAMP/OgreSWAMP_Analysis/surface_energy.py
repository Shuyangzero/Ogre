

#from ogre.utils import convergence_plots


convergence_plots_kw = \
{
    "structure_name": "ASPIRIN",
    "scf_path": "SCF",
    "threshold":  5,
    ## MAYBE MAX_LAYERS INSTEAD
    "consecutive_step": 1,
    "fontsize": 16,
}

convergence_plots(**convergence_plots_kw)