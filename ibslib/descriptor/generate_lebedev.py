# -*- coding: utf-8 -*-
import json
import numpy as np
import spaudiopy as spa

azi, colat, weights = spa.grids.lebedev(n=64)
coords = spa.utils.sph2cart(azi, colat)
coords = np.hstack([coords[0][:,None], 
                    coords[1][:,None], 
                    coords[2][:,None]])

temp_dict = {"coords": coords.tolist()}

with open("lebedev.py", "a+") as f:
    f.write(json.dumps(temp_dict))