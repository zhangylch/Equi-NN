import torch
from torch import nn
import numpy as np

class SPH_CAL(nn.Module):
    def __init__(self,max_l=3):
        '''
         max_l: maximum L for angular momentum

        '''
        super().__init__()
        #  form [0,max_L]
        if max_l<1: raise ValueError("The angular momentum must be greater than or equal to 1. Or the angular momentum is lack of angular information, the calculation of the sph is meanless.")
        self.register_buffer("max_l",torch.tensor([int(max_l+1)],dtype=torch.long))
        self.register_buffer("dim",torch.tensor([min(9,(max_l+1)*(max_l+1))],dtype=torch.long))
        pt=torch.empty((self.max_l[0],self.max_l[0]),dtype=torch.long)
        pt_rev=torch.empty((self.max_l[0],self.max_l[0]),dtype=torch.long)
        yr=torch.empty((self.max_l[0],self.max_l[0]),dtype=torch.long)
        yr_rev=torch.empty((self.max_l[0],self.max_l[0]),dtype=torch.long)
        num_lm=int((self.max_l[0]+1)*self.max_l[0]/2)
        coeff_a=torch.empty(num_lm)
        coeff_b=torch.empty(num_lm)
        tmp=torch.arange(self.max_l[0],dtype=torch.long)
        self.register_buffer("prefactor1",-torch.sqrt(1.0+0.5/tmp))
        self.register_buffer("prefactor2",torch.sqrt(2.0*tmp+3))
        ls=tmp*tmp
        for l in range(self.max_l[0]):
            pt[l,0:l+1]=tmp[0:l+1]+int(l*(l+1)/2)
            pt_rev[l,1:l]=l-tmp[1:l]+int(l*(l+1)/2)
            # here the self.yr and self.yr_rev have overlap in m=0.
            yr[l,0:l+1]=ls[l]+l+tmp[0:l+1]
            yr_rev[l,0:l+1]=ls[l]+tmp[0:l+1]
            if l>0.5:
                coeff_a[pt[l,0:l]]=torch.sqrt((4.0*ls[l]-1)/(ls[l]-ls[0:l]))
                coeff_b[pt[l,0:l]]=-torch.sqrt((ls[l-1]-ls[0:l])/(4.0*ls[l-1]-1.0))
          
        self.register_buffer("pt",pt)
        self.register_buffer("pt_rev",pt_rev)
        self.register_buffer("yr",yr)
        self.register_buffer("yr_rev",yr_rev)
        self.register_buffer("coeff_a",coeff_a)
        self.register_buffer("coeff_b",coeff_b)

        self.register_buffer("sqrt2_rev",torch.sqrt(torch.tensor([1/2.0])))
        self.register_buffer("sqrt2pi_rev",torch.sqrt(torch.tensor([0.5/np.pi])))
        self.register_buffer("hc_factor1",torch.sqrt(torch.tensor([15.0/4.0/np.pi])))
        self.register_buffer("hc_factor2",torch.sqrt(torch.tensor([5.0/16.0/np.pi])))
        self.register_buffer("hc_factor3",torch.sqrt(torch.tensor([15.0/16.0/np.pi])))


    def forward(self,cart):
        distances=torch.linalg.norm(cart,dim=0)  # to convert to the dimension (n,batchsize)
        d_sq=distances*distances
        sph_shape=(int(self.dim[0]),)+cart.shape[1:]
        sph=cart.new_empty(sph_shape,device=cart.device)
        sph[0]=self.sqrt2pi_rev*self.sqrt2_rev
        sph[1]=self.prefactor1[1]*self.sqrt2pi_rev*cart[1]
        sph[2]=self.prefactor2[0]*self.sqrt2_rev*self.sqrt2pi_rev*cart[2]
        sph[3]=self.prefactor1[1]*self.sqrt2pi_rev*cart[0]
        if self.max_l[0]>2.5:
            sph[4]=self.hc_factor1*cart[0]*cart[1]
            sph[5]=-self.hc_factor1*cart[1]*cart[2]
            sph[6]=self.hc_factor2*(3.0*cart[2]*cart[2]-d_sq)
            sph[7]=-self.hc_factor1*cart[0]*cart[2]
            sph[8]=self.hc_factor3*(cart[0]*cart[0]-cart[1]*cart[1])
            for l in range(3,int(self.max_l[0])):
                sph1=self.prefactor1[l]*(cart[0:1]*sph[self.yr_rev[l-1,0]]+cart[1:2]*sph[self.yr[l-1,l-1]])
                sph2=self.prefactor2[l-1]*cart[2:3]*sph[self.yr_rev[l-1,0]]
                sph3=torch.einsum("i,i... ->i...",self.coeff_a[self.pt_rev[l,2:l]],(cart[2]*sph[self.yr_rev[l-1,1:l-1]]+torch.einsum("i,...,i... ->i...",self.coeff_b[self.pt_rev[l,2:l]],d_sq,sph[self.yr_rev[l-2,0:l-2]])))
                sph4=torch.einsum("i,i...->i...",self.coeff_a[self.pt[l,0:l-1]],(cart[2]*sph[self.yr[l-1,0:l-1]]+torch.einsum("i,...,i... ->i...",self.coeff_b[self.pt[l,0:l-1]],d_sq,sph[self.yr[l-2,0:l-1]])))
                sph5=self.prefactor2[l-1]*cart[2:3]*sph[self.yr[l-1,l-1]]
                sph6=self.prefactor1[l]*(cart[0:1]*sph[self.yr[l-1,l-1]]-cart[1:2]*sph[self.yr_rev[l-1,0]])
                sph=torch.vstack((sph,sph1,sph2,sph3,sph4,sph5,sph6))
        return sph
'''
# here is an example to use the sph calculation
import timeit 
import torch._dynamo
import sph_cal as sph_test
max_l=16
cart=torch.randn((3,1000000),dtype=torch.float32)
sph_cal=SPH_CAL(max_l=max_l)
sph_bench=sph_test.SPH_CAL(max_l=max_l)
#torch._dynamo.config.verbose=True
#torch._dynamo.config.suppress_errors = True
sph_cal(cart)
sph_bench(cart)
starttime = timeit.default_timer()
tmp=sph_cal(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
starttime = timeit.default_timer()
tmp1=sph_bench(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp-tmp1)
compile_sph=torch.compile(sph_cal,mode="max-autotune")
x=compile_sph(cart)
starttime = timeit.default_timer()
tmp=compile_sph(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
compile_sph=torch.compile(sph_bench,mode="max-autotune")
x=compile_sph(cart)
starttime = timeit.default_timer()
tmp=compile_sph(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
#forward=vmap(sph_cal,in_dims=2,out_dims=2)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=forward(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#compile_sph=torch.compile(forward,mode="max-autotune")
#x=compile_sph(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=compile_sph(cart)
#print("The time difference is :", timeit.default_timer() - starttime)

#jac=jax.jit(jax.vmap(jax.jacfwd(test_forward),in_axes=(1),out_axes=(1)))
#grad=jac(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#grad=jac(cart)
#print("The time difference is :", timeit.default_timer() - starttime)

#hess=jax.jit(jax.vmap(jax.hessian(sph.compute_sph),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#
## calculate hessian by jac(jac)
#hess=jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(sph.compute_sph)),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#print(tmp.shape)
'''
