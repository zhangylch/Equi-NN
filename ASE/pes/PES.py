import torch
import numpy as np
import os
from src.params import *
import model.MPNN as MPNN

class PES(torch.nn.Module):
    def __init__(self):
        super(PES, self).__init__()
        
        self.model=MPNN.MPNN(100,atom_species=atom_species,max_l=max_l,nwave=nwave,cutoff=cutoff,ncontract=ncontract,emb_nblock=emb_nblock,emb_nl=emb_nl,emb_layernorm=emb_layernorm,iter_loop=iter_loop,iter_nblock=iter_nblock,iter_nl=iter_nl,iter_dropout_p=iter_dropout_p,iter_layernorm=iter_layernorm,nblock=nblock,nl=nl,dropout_p=dropout_p,layernorm=layernorm,Dtype=torch_dtype).to(device).to(torch_dtype)
     
    def forward(self,cell,disp_cell,cart,neigh_list,shifts,species):
        symm_cell=(disp_cell+disp_cell.permute(1,0))/2.0
        cart=cart+torch.einsum("jk,km ->jm",cart,symm_cell)
        cell=cell+torch.einsum("jk,km -> jm",cell,symm_cell)
        shifts=torch.einsum("jk,km ->jm",shifts,cell)
        center_factor=cart.new_ones(cart.shape[0])
        neigh_factor=cart.new_ones(neigh_list.shape[1])
        energy=self.model(cart,neigh_list,shifts,center_factor,neigh_factor,species)
        return energy
