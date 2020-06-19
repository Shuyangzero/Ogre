
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



print(wulffmaker_index(indices))
print(wulffmaker_color(indices))
print(wulffmaker_gamma(energies))

miller_index_legend(sort_keys(indices))

