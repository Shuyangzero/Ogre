# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='ogre',
    version='0.0',
    packages=['ogre',
              'ogre/ogreSWAMP',
              'ogre/utils',
              'ibslib',
              'ibslib/analysis',
              'ibslib/crossover',
              'ibslib/descriptor',
              'ibslib/io',
              'ibslib/molecules',
              'ibslib/motif',
              'ibslib/structures',
              'ibslib/sgroup',
              'ibslib/database',
              'ibslib/calculators',
              'ibslib/libmpi',
              'ibslib/report',
              'ibslib/plot',
              'ibslib/acsf'
              ],
    # find_packages(exclude=[]),
    install_requires=['numpy', 'matplotlib', 'pymatgen==2019.6.5', "sklearn",
                      "torch", "scipy", "pymongo", "pandas",
                      'networkx==2.3', 'tqdm',
                      "ase"],
    include_package_data=True,
)
