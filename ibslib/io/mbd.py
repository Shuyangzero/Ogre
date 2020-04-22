


"""
Writing XYZ style files for running the MBD calculation using the code 
provided at: http://th.fhi-berlin.mpg.de/~tkatchen/MBD.
"""

from ibslib import Structure

def xyz_mbd_str(struct):
    """
    Returns the string to be written to a file for the MBD calculations.

    Arguments
    ---------
    struct: Structure
        Structure to produce the string for. Must have "hirshfeld_volumes"
        in the properties of the Structure. 
    """
    if type(struct) != Structure:
        raise Exception("Argument to xyz_mbd_str was not a Structure object.")
    
    volumes = struct.get_property("hirshfeld_volumes")
    geo = struct.get_geo_array()
    elements = struct.geometry["element"]
    lattice = struct.get_lattice_vectors_better()

    if volumes == None:
        raise Exception("Structure argument to xyz_mbd_str does not contain "+
                "hirshfeld_volumes in its properties."+
                "Please extract these from FHI-aims calculation.")
    if len(volumes) != geo.shape[0]:
        raise Exception("Error in writing mbd files. "+
                "Number of hirshfeld_volumes was {} and the number "
                .format(len(volumes)) +
                "of atoms was {}. These values must be the same."
                .format(geo.shape[0]))

    file_str = ""
    file_str += str(geo.shape[0])
    file_str += "\n\n"

    for i,ele in enumerate(elements):
        line = ""
        line += str(ele) + " "
        for coord in geo[i,:]:
            line += str(coord) + " "
        
        line += str(volumes[i]) + "\n"
        file_str += line
    
    if len(lattice) != 0:
        for row in lattice:
            file_str += "lattice_vector "
            for value in row:
                file_str += str(value) + " "
            file_str += "\n"
    
    return file_str


