import torch
from src.params import *
import lammps_MPNN.model.MPNN as MPNN
from collections import OrderedDict

class lammps():
    def __init__(self,atom_species=torch.tensor(np.array([[1]])),initpot=0.0,max_l=2,nwave=8,cutoff=4.0,ncontract=64,emb_nblock=1,emb_nl=[8,8],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64],dropout_p=[0.0,0.0],layernorm=True):
        self.model=MPNN.MPNN(atom_species=atom_species,initpot=0.0,max_l=max_l,nwave=nwave,cutoff=cutoff,ncontract=ncontract,emb_nblock=emb_nblock,emb_nl=emb_nl,emb_layernorm=emb_layernorm,iter_loop=iter_loop,iter_nblock=iter_nblock,iter_nl=iter_nl,iter_dropout_p=iter_dropout_p,iter_layernorm=iter_layernorm,nblock=nblock,nl=nl,dropout_p=dropout_p,layernorm=layernorm).to(torch.device("cpu"))
    
    def __call__(self,ema_model):    
        state_dict = ema_model.state_dict()
        self.model.load_state_dict(state_dict)
        scripted_pes=torch.jit.script(self.model)
        scripted_pes.save("LAMMPS.pt")
