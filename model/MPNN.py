import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like,Tanh_like

class MPNN(torch.nn.Module):
    def __init__(self,neigh_atoms,atom_species=torch.tensor(np.array([[1]])),initpot=0.0,max_l=2,nwave=8,cutoff=4.0,ncontract=64,emb_nblock=1,emb_nl=[1,8,8],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64,64],dropout_p=[0.0,0.0],layernorm=True,Dtype=torch.float32):
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
        self.index_l=index_l

        initbias=torch.randn(nwave,dtype=Dtype)/neigh_atoms
        alpha=torch.ones(nwave,dtype=Dtype)
        rs=(torch.rand(nwave)*np.sqrt(cutoff)).to(Dtype)
        initbias=torch.hstack((initbias,alpha,rs))
        # embedded nn
        self.emb_neighnn=MLP.NNMod(nwave*3,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)
        initbias=torch.randn(ncontract,dtype=Dtype)/neigh_atoms
        self.emb_centernn=MLP.NNMod(ncontract,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)

        # instantiate the nn radial function and disable the dropout 
        self.sph_cal=sph_cal.SPH_CAL(max_l)
        itermod=OrderedDict()
        for i in range(iter_loop):
            initbias=torch.randn(nwave,dtype=Dtype)
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(nwave,iter_nblock,iter_nl,iter_dropout_p,Relu_like,initbias=initbias,layernorm=iter_layernorm)
        self.itermod= torch.nn.ModuleDict(itermod)
        self.outnn=MLP.NNMod(1,nblock,nl,dropout_p,Relu_like,initbias=torch.tensor(np.array([initpot])),layernorm=layernorm)

    def forward(self,cart,neighlist,shifts,center_factor,neigh_factor,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        center_coeff=self.emb_centernn(species)
        neigh_coeff=self.emb_neighnn(species)
        neigh_emb=neigh_coeff[neighlist[1]].T.contiguous()
        cut_distances=neigh_factor*self.cutoff_cosine(distances)
        # for the efficiency of traditional ANN, we do the first calculation of density mannually.
        radial_func=torch.exp(-torch.square(neigh_emb[self.nwave:self.nwave*2]*(distances-neigh_emb[self.nwave*2:])))
        contracted_coeff=self.contracted_coeff[:,self.index_l].contiguous()
        sph=self.sph_cal(distvec.T)
        orbital=torch.einsum("i,ji,ji,ki->ikj",cut_distances,radial_func,neigh_emb[:self.nwave],sph)
        center_orbital=cart.new_zeros((cart.shape[0],self.nangular,self.nwave),dtype=cart.dtype,device=cart.device)
        center_orbital=torch.index_add(center_orbital,0,neighlist[0],orbital)
        contracted_orbital=torch.einsum("ikj,kjm->ikm",center_orbital,contracted_coeff[0])
        density=torch.einsum("ikm,ikm,im ->im",contracted_orbital,contracted_orbital,center_coeff)
        for iter_loop, (_, m) in enumerate(self.itermod.items()):
            iter_coeff=m(density)
            iter_density,center_orbital=self.density(orbital,cut_distances,iter_coeff,neighlist[0],neighlist[1],contracted_coeff[iter_loop+1],center_orbital,center_coeff)
            density=density+iter_density
            # here cente_coeff is for discriminating for the different center atoms.
        output=self.outnn(density)
        return torch.einsum("ij,i ->",output,center_factor)

    def density(self,orbital,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,center_orbital,center_coeff):
        weight_orbital=torch.einsum("ij,ikj -> ikj",iter_coeff[index_neigh],orbital)+torch.einsum("ikj,i->ikj",center_orbital[index_neigh],cut_distances)
        center_orbital=torch.index_add(center_orbital,0,index_center,weight_orbital)
        contracted_orbital=torch.einsum("ikj,kjm->ikm",center_orbital,contracted_coeff)
        density=torch.einsum("ikm,ikm,im ->im",contracted_orbital,contracted_orbital,center_coeff)
        return density,center_orbital
     
    def cutoff_cosine(self,distances):
        tmp=0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5
        return tmp*tmp

    def radial_func(self,distances,alpha,rs):
        return torch.exp(-torch.square(alpha*(distances[:,None]-rs)))
