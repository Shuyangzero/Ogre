# Ogre
## Basic Command
Create organic interfaces:
`python main.py --path test/organic/`

Create organic Interfaces with repaired molecules:
`python main.py --path test/organic/ --repair True`

Create inorganic interfaces:
`python main.py --path test/inorganic/`
## Illustration
In the code, there is an option `repair`. To make it clear, some illustrstions are needed.  

As we all known, the molecules will break when users make a slab, since the boundaries of slab would mostly go across the molecules that leave some but not all atoms in the boundary.
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/raw_QQQCIG04_O.png)
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/100_face_1layer_unrepaired_QQQCIG04_O.png)
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/111_face_3layers_unrepaired.png)
In order to precisely describe the physical structure and the properties of slabs. We generate two methods for you to take.
First of all, the default one, you could delete the broken molecules. This methods means that once we detact that the molecules are broken, we would delete it and return the rest of them.
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/100_face_1layer_clean_QQQCIG04_O.png)
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/111_face_3layers_clean_QQQC1G04_O.png)
The good thing is that even though some of the molecules disappeare, the physical structure and the electronic properties are basically maintained and this structure could be used to calculate DFT or other calculations that the load are sensitive to the amount of atoms.  

Secondly, the optional one, you could repair the broken molecules by taking `--repair True`. This method could hopefully find the connection between broken molecule and the correlated intact one, and replace them with the intact molecules.
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/100_face_1layer_repaired_QQQCIG04_O.png)
![image](https://github.com/Shuyangzero/LatUnmatchIB/blob/master/raw_picture/111_face_3layers_repaired.png)
The pros is that this method would maintain the physical structure and properties as much as it can. However, it would bring extra loads to quantum calculations and sometimes takes way more time. What's more, the algorithm for repairing broken molecules is based on enough information, which means that those molecules which just have two or three bonds, could not be repaired. Additionally, this method would delete the broken molecules on the -c direction and repair the brokens on the +c direction, in the wish of keeping the amount of atoms in the slab. For those slabs that has no intact molecules, this methods would hopefully work and repair all broken molecules.  

All in all, deleting broken molecules would be better to quantum calculations, and repairing broken molecules would be better to properties' exploration.
