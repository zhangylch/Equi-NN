# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen


modified by Yaolong Zhang for a better efficiency
"""

import torch
import ase.io.vasp
from ase import Atoms, units
import getneigh as getneigh
from ase.calculators.Equi_MPNN import Equi_MPNN
from ase.io import extxyz
from ase.io.trajectory import Trajectory
import time

from ase.optimize import BFGS,FIRE
from ase.constraints import ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
from ase.md.nvtberendsen import NVTBerendsen 
from ase.md import MDLogger
import numpy as np

ef=np.zeros(3)
fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,1))
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
device='cpu'
maxneigh=25000# maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
calc=Equi_MPNN(maxneigh, getneigh, potential = "PES.pt", device=device, dtype = torch.float32)
start=time.time()
num=0.0
for atoms in configuration:
    calc.reset()
    atoms.calc=calc
    traj = Trajectory('co+water+au_nvt.traj', 'w', atoms)
    MBD(atoms,temperature_K=300)
    dyn = Langevin(atoms, 0.1*units.fs, temperature_K=300, friction=5e-3)
    dyn.attach(traj.write, interval=10)
    dyn.attach(MDLogger(dyn,atoms,'md_nvt.log',header=True,mode='w'), interval=10)
    dyn.run(steps=100000)
    traj.close()
