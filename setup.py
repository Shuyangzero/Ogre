# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='ogreswamp',
    version='0.0',
    packages=['ogreswamp',
              ],
    #find_packages(exclude=[]),
    install_requires=['numpy', 'matplotlib', 'pymatgen==2019.6.5', 'networkx==2.3','tqdm',
                      "ase @ https://gitlab.com/ase/tarball/master"],
)
