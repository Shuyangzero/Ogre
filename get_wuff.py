from ibslib.io import read
from pymatgen.io.cif import CifParser
from pymatgen import Lattice
from pymatgen.analysis.wulff import WulffShape
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
with open("surface_energy.pickle",'rb') as file:
    energy_results = pickle.load(file)
cif = CifParser("bulk.cif")
lattice = cif.get_structures()[0].lattice
print(lattice.a, lattice.b, lattice.c)
for tag in ["ts", "mbd"]:
    energy_result = energy_results[tag]
    data = {}
    for index, term, by, ly in energy_result:
        idx = []
        temp_idx = ""
        for char in index:
            if char == "-":
                temp_idx += char
            else:
                temp_idx += char
                idx.append(temp_idx)
                temp_idx = ""
        idx = tuple([int(x) for x in idx])
        if idx not in data or data[idx] > ly:
            data[idx] = float(ly)
    print(tag)
    print(data.keys(), data.values())
    plt.figure()
    w_r = WulffShape(lattice, data.keys(), data.values())
    print("plot")
    w_r.get_plot(direction=(0,1,0))
    plt.savefig("wuff_{}.png".format(tag),dpi=400, bbox_inches="tight")