#!/usr/bin/python

from ase.io.trajectory import TrajectoryReader
from ase.io import write
import numpy as np
import os

#reading in the trajectory file created during 
#optimization
traj = TrajectoryReader("co+water+au_nvt.traj")
outfile=open("traj1.extxyz",'w')
#write each structure from the .traj file in .xyz format
for atoms in traj: 
    #string = 'structure%03d' % (i,) +'.xyz'
    write(outfile, atoms, format="extxyz")
    coor=atoms.get_positions()
    print(np.sqrt(np.sum(np.square(coor[144]-coor[145]))))
outfile.close()
traj.close()
