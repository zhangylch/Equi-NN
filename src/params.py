import time
import torch
import numpy as np
import os
from src.gpu_sel import *

# open a file for output information in iterations
fout=open('nn.err','w')
#======general setup===========================================
table_init=0                   # 1: a pretrained or restart  
force_table=True
ratio=0.9                      # ratio for vaildation
find_unused = False
queue_size=10
Epoch=50000                    # total numbers of epochs for fitting 
dtype='float32'   #float32/float64
# batchsize: the most import setup for efficiency
batchsize=64  # batchsize for each process
init_weight=[1.0, 5.0]
final_weight=[1.0, 0.5]
ema_decay=0.999
check_epoch=10

#========================parameters for optim=======================
start_lr=0.01                  # initial learning rate
end_lr=1e-5                    # final learning rate
re_coeff=0.0                    # L2 normalization cofficient
decay_factor=0.5               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
patience_epoch=100             # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
datafloder="./"

#=======================parameters for local environment========================
maxneigh=100000
cutoff = 4.0
max_l=2
nwave=8
ncontract=None

#===============================embedded NN structure==========
emb_nblock=1
emb_nl=[8,8]
emb_layernorm=True


iter_loop = 2
iter_nblock = 1             # neural network architecture   
iter_nl = [64,64]
iter_dropout_p=[0.0,0.0,0.0,0.0]
iter_layernorm=False

#======== parameters for final output nn=================================================
nblock = 1                     # the number of resduial NN blocks
nl=[64,64]                   # NN structure
dropout_p=[0.0,0.0]            # dropout probability for each hidden layer
layernorm = False

#======================read input=================================================================
with open('para/input','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])


# add the input neuron for each neuron
emb_nl.insert(0,1)
iter_nl.insert(0,ncontract)
nl.insert(0,ncontract)

# torch and numpy dtype
if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32


torch.set_default_dtype(torch_dtype)

if not ncontract: ncontract=int(nwave*(nwave+1)/2)

gpu_sel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==============================train data loader===================================
init_weight=torch.tensor(init_weight).to(torch_dtype).to(device)
final_weight=torch.tensor(final_weight).to(torch_dtype).to(device)
if force_table:
    nprop=2
else:
    nprop=1
    init_weight=init_weight[0:1]
    final_weight=final_weight[0:1]

atom_species=torch.tensor(np.array(atom_species)).reshape(-1,1).to(torch_dtype).to(device)

