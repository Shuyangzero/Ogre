

Ogre
====

..
    .. toctree::
       :maxdepth: 2
       :numbered:


Introduction
------------

Ogre is written in Python and interfaces with the FHI-aims code to calculate surface energies at the level of density functional theory (DFT). The input of Ogre is the geometry of the bulk molecular crystal. The surface is cleaved from the bulk structure with the molecules on the surface kept intact. A slab model is constructed according to the user specifications for the number of molecular layers and the length of the vacuum region. Ogre automatically identifies all symmetrically unique surfaces for the user-specified Miller indices and detects all possible surface terminations. Ogre includes utilities, OgreSWAMP, to analyze the surface energy convergence and Wulff shape of the molecular crystal.


Installation
------------

DO THIS

Examples
--------

The examples are organized as follows

Ogre
^^^^

OgreSWAMP
^^^^^^^^^


Unique Surfaces
~~~~~~~~~~~~~~~

Include the algorithm flow chart. 


Surface energy and convergence plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain example here. Include an explanation of the features of the generated
images. 


Wulffmaker
~~~~~~~~~~

This will describe how this code utilizes Wulffmaker. Include images of where each string should be pasted into the Wulffmaker source code. Include a description of the legend that is generated. Include description of the str2tuple function. Use Aspirin as example, so include lattice and space group information.  

Code Reference
--------------

This section is organized as follows

Ogre
^^^^

.. autoclass:: ogre.generators.OrganicSlabGenerator
    :members: cleave

.. autofunction:: ogre.generators.cleave_for_surface_energies

.. autofunction:: ogre.generators.atomic_task



OgreSWAMP
^^^^^^^^^

.. autoclass:: ogre.utils.UniquePlanes
    :members: prep_idx, get_cell, get_hall_number, miller_to_real, real_to_miller, find_unique_planes
    
.. autofunction:: ogre.utils.convergence_plots

.. autofunction:: ogre.utils.wulffmaker.wulffmaker_index

.. autofunction:: ogre.utils.wulffmaker.wulffmaker_gamma

.. autofunction:: ogre.utils.wulffmaker.wulffmaker_color

.. autofunction:: ogre.utils.wulffmaker.miller_index_legend

    
