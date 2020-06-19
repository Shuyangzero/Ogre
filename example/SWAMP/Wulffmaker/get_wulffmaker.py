
import json
from ogre.utils.wulffmaker import sort_keys,str2tuple,\
                                 wulffmaker_index, wulffmaker_gamma, \
                                 wulffmaker_color, \
                                 miller_index_legend
                                 
"""
Example of how to use functions in ogre.utils.wulffmaker to automatically 
generate data a Wulffmaker format.

To get information about any of the functions, you may type, for example,
help(wulffmaker_index)

This example uses Aspirin. The bulk structure of Aspirin has space group 
symmetry P2_1/c making it monoclinic with 2/m type symmetry. The lattice 
parameters are (11.233,6.5440,11.231,90,95.89,90). This information may be 
added to the default values in the Wulffmaker source code or can be edited 
in the graphical user interface. 

"""

result_file = "results.json"
method = "mbd"

with open("results.json") as f:
    results_dict = json.load(f)
    
results = results_dict[method]
indices = []
energies = []

for entry in results:
    indices.append(str2tuple(entry[0]))
    energies.append(entry[-1])


print("------------------------------------")
print(wulffmaker_index(indices))
print("------------------------------------")
print(wulffmaker_color(indices))
print("------------------------------------")
print(wulffmaker_gamma(energies))
print("------------------------------------")

miller_index_legend(sort_keys(indices))


















