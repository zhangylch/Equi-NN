#----------------fireann interface is for Equi-MPNN package-------------------------


import numpy as np
import os
import torch
import re
#from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)

class Equi_MPNN(Calculator):

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, maxneigh, getneigh, nn = 'PES.pt',device="cpu",dtype=torch.float32,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.maxneigh=maxneigh
        self.getneigh=getneigh
        pes=torch.jit.load(nn,map_location=torch.device(device))
        pes.to(self.device).to(self.dtype)
        pes.eval()
        self.cutoff=pes.model.cutoff
        self.pes=torch.jit.optimize_for_inference(pes)
        self.tcell=[]
        #self.pes=torch.compile(pes)
    
    def calculate(self,atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        cell=np.array(self.atoms.cell)
        if "cell" in system_changes:
            if cell.ndim==1:
                cell=np.diag(cell)
            self.getneigh.init_neigh(self.cutoff,self.cutoff/2.0,cell.T)
            self.tcell=torch.from_numpy(cell).to(self.dtype).to(self.device)
        icart = self.atoms.get_positions()
        cart,neighlist,shiftimage,scutnum=self.getneigh.get_neigh(icart.T,self.maxneigh)
        cart=torch.from_numpy(cart.T).contiguous().to(self.device).to(self.dtype)
        neighlist=torch.from_numpy(neighlist[:,:scutnum]).contiguous().to(self.device).to(torch.long)
        shifts=torch.from_numpy(shiftimage.T[:scutnum,:]).contiguous().to(self.device).to(self.dtype)
        species = self.atoms.get_atomic_numbers()
        species = torch.tensor(np.array(species),device=self.device,dtype=self.dtype).view(-1,1)
        disp_cell = torch.zeros_like(self.tcell)

        if "forces" in properties:
            cart.requires_grad=True
        else:
            cart.requires_grad=False

        if "stress" in properties:
            disp_cell.requires_grad=True
        else:
            disp_cell.requires_grad=False
        energy=self.pes(self.tcell,disp_cell,cart,neighlist,shifts,species)
        self.results['energy'] = float(energy.detach().numpy())
        if "forces" in properties and "stress" in properties:
            forces,virial = torch.autograd.grad(energy,[cart,disp_cell])
            forces = torch.neg(forces).squeeze(0).detach().numpy()
            self.results['forces'] = forces
            virial = virial.squeeze(0).detach().numpy()
            self.results['stress'] = virial/self.atoms.get_volume()

        if "forces" in properties and "stress" not in properties:
            forces = torch.autograd.grad(energy,cart)[0].squeeze(0)
            forces = torch.neg(forces).detach().numpy()
            self.results['forces'] = forces

        if "stress" in properties and "forces" not in properties:
            virial = torch.autograd.grad(energy,disp_cell)[0].squeeze(0)
            virial = virial.detach().numpy()
            self.results['stress'] = virial/get_volume()

