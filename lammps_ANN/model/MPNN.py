import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like,Tanh_like

class MPNN(torch.nn.Module):
    def __init__(self,atom_species=torch.tensor(np.array([[1]])),initpot=0.0,max_l=2,nwave=8,cutoff=4.0,ncontract=64,emb_nblock=1,emb_nl=[1,8,8],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64,64],dropout_p=[0.0,0.0],layernorm=True):
        super(MPNN,self).__init__()
        rmaxl=max_l+1 
        self.nangular=rmaxl*rmaxl
        self.nwave=nwave
        self.cutoff=cutoff
        self.register_buffer("atom_species",atom_species)

        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(iter_loop+1,rmaxl,nwave,ncontract)))

        index_l=torch.zeros(self.nangular,dtype=torch.long)
        for l in range(rmaxl):
            index_l[l*l:(l+1)*(l+1)]=l           

        self.register_buffer("index_l",index_l)

        initbias=torch.randn(nwave)
        alpha=torch.ones(nwave)
        rs=(torch.rand(nwave)*np.sqrt(cutoff))
        initbias=torch.hstack((initbias,alpha,rs))
        # embedded nn
        self.emb_neighnn=MLP.NNMod(nwave*3,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)
        initbias=torch.randn(ncontract)
        self.emb_centernn=MLP.NNMod(ncontract,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)

        # instantiate the nn radial function and disable the dropout 
        self.sph_cal=sph_cal.SPH_CAL(max_l)
        self.outnn=MLP.NNMod(1,nblock,nl,dropout_p,Relu_like,initbias=torch.tensor(np.array([initpot])),layernorm=layernorm)

    def forward(self,cart,centerlist,neighlist,local_species,neigh_species,nlocal):
        cart.requires_grad_(True)
        distvec=cart[neighlist]-cart[centerlist]
        distances=torch.linalg.norm(distvec,dim=1)
        local_coeff=self.emb_centernn(self.atom_species)
        neigh_coeff=self.emb_neighnn(self.atom_species)
        neigh_emb=(neigh_coeff[local_species]).T.contiguous()
        cut_distances=torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)
        radial_func=torch.exp(-torch.square(neigh_emb[self.nwave:self.nwave*2]*(distances-neigh_emb[self.nwave*2:])))
        contracted_coeff=self.contracted_coeff[:,self.index_l]
        sph=self.sph_cal(distvec.T)
        orbital=torch.einsum("i,ji,ji,ki->ikj",cut_distances,radial_func,neigh_emb[:self.nwave],sph)
        center_orbital=cart.new_zeros((nlocal.shape[0],self.nangular,self.nwave),dtype=cart.dtype,device=cart.device)
        center_orbital=torch.index_add(center_orbital,0,centerlist,orbital)
        contracted_orbital=torch.einsum("ikj,kjm->ikm",center_orbital,contracted_coeff[0])
        density=torch.einsum("ikm,ikm,im ->im",contracted_orbital,contracted_orbital,local_coeff)
        output=self.outnn(density)
        energy=torch.sum(output)
        force=torch.autograd.grad([energy,],[cart,])[0]
        if force is not None:
            return energy.detach(),-force.reshape(-1).detach(),output.detach()

