
import json
from ogre.utils import convergence_plots

convergence_plots_kw = \
{
    "structure_name": "ASPIRIN",
    "scf_path": "SCF",
    "threshold":  5,
    "max_layers": -1,
    "fontsize": 16,
    "pbe": True,
    "mbd": True,
    "boettger": True,
    "combined_figure": True
}
results_file = "results.json"

results = convergence_plots(**convergence_plots_kw)

with open(results_file,"w") as f:
    f.write(json.dumps(results, indent=4))
    
