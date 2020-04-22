


from ibslib.io import aims_extractor

__author__='Manny Bier'

def extract(struct_dir, extractor="aims", extractor_kwargs={}):
    """ 
    
    Purpose is to extract information from a specific direcoty format.
        For example, extract FHI-aims calculation directories to a Structure
        json file. 
    
    Arguments
    ---------
    struct_dir: path
        Path to the directory that information will be extracted from
    extractor: str
        Extraction method to use
    kwargs: dict
        Dictionary of keyword arguments which will be passed to the extraction
        process.
    
    """
    
    if extractor == "aims":
        result = aims_extractor.extract(struct_dir, extractor_kwargs)
    
    return result
        





if __name__ == "__main__":
    struct_dir = "/Users/ibier/Research/Results/Hab_Project/FUQJIK/2_mpc/Genarris/Relaxation"
    result = extract(struct_dir, extractor="aims")
