# -*- coding: utf-8 -*-

'''
Purpose:
    This will be a master compute descriptor file that the user interacts with.
    It will be able to compute the descriptors for an entire pool of structures 
    while importing modules on the fly. It will call the appropriate files
    for the users. 
'''

def compute_descriptor_dict(struct_dict, descriptor='rdf', **kwargs):
    '''
    Usage:
        Pass in all settings for the descriptor calculations in **kwargs
    '''
    module_name = 'descriptor.{}'.format(descriptor)
    descriptor_module = my_import(module_name,package='ibslib')
    descriptor_module.eval_dict(struct_dict, **kwargs)
    return struct_dict    


def my_import(name, package=''):
    '''
    dynamically (at runtime) imports modules portentially specified in the UI
    taken from http://function.name/in/Python/__import__ 
    '''
    name = package + '.' + name
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod 