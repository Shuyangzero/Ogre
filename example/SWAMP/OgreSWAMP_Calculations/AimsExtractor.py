# -*- coding: utf-8 -*-



from ibslib.io import read,write,AimsExtractor
from ibslib.io.aims_extractor import name_abs_path,name_from_path

# Input directory to extract from
calc_dir = "batch_calc"
# Output directory for structure files
output_dir = "SCF"
# Type of output file
output_format = "json"
# All properties: "energy","vdw_energy","time","space_group","hirshfeld_volumes"
aims_property = ["energy", "hirshfeld_volumes", "vdw_energy","time"]
# Name for stored total energy value
energy_name = "energy"

# Definition of keyword arguments dictionary to be passed to extractor
extractor_kw = \
    {
        "aims_property": aims_property,
        "energy_name": energy_name,
    }

# Extractor returns StructDict
ae = AimsExtractor(calc_dir, **extractor_kw, symprec=0.1)
struct_dict = ae.run_extraction()


### Make naming compatable with Genarris
for struct_id,struct in struct_dict.items():
    struct.properties["energy"] = struct.properties[energy_name]
    struct.get_unit_cell_volume()

write(output_dir, struct_dict, file_format=output_format,
      overwrite=True)
