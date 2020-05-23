

#from ogre.utils import convergence_plots


convergence_plots_kw = \
{
    "structure_name": "ASPIRIN",
    "scf_path": "SCF",
    "threshold":  1,
    ## MAYBE MAX_LAYERS INSTEAD
    "max_layers": -1,
    "fontsize": 16,
}

convergence_plots(**convergence_plots_kw)