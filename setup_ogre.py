# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='ogre',
    version='0.0',
    packages=['ogre',
              'ogre/utils',
              #'ibslib',
              #'ibslib/analysis',
              #'ibslib/io',
              #'ibslib/structures',
              #'ibslib/calculators',
              ],
    # find_packages(exclude=[]),
    install_requires=['numpy', 'matplotlib', 'pymatgen==2019.6.5', "sklearn",
                      "torch", "scipy", "pymongo", "pandas",
                      'networkx==2.3', 'tqdm',
                      "ase"],
    include_package_data=True,
)
