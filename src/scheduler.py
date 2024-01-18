import torch
import numpy as np

# define the strategy of weight decay
class Scheduler():
    def __init__(self,end_lr,decay_factor,state_loader,optim,model,ema_model,jit_lammps):
        self.best_loss = 1e30
        self.end_lr=end_lr
        self.decay_factor=decay_factor
        self.state_loader=state_loader
        self.optim=optim
        self.ema_model=ema_model
        self.model=model
        self.jit_lammps=jit_lammps
    
    def __call__(self,loss):
        return self.forward(loss)
 
    def forward(self,loss):
        if loss>10.0*self.best_loss or loss.isnan():
            lr=self.optim.param_groups[0]["lr"]
            self.state_loader(self.model,"Equi-MPNN.pt")
            self.state_loader(self.ema_model,"ema.pt")
            self.optim.param_groups[0]["lr"]=lr*self.decay_factor
        else:
            # store the best loss for preventing a boomm of error
            if loss<self.best_loss:
                self.best_loss = loss
                # save the jit model for inference
                jit_pes=torch.jit.script(self.ema_model.module)
                jit_pes.save("inference.pt")
                # store the checkpoint at each epoch
                torch.save(self.ema_model.state_dict(),"ema.pt")
                torch.save(self.model.state_dict(),"Equi-MPNN.pt")
                torch.save(self.optim.state_dict(),"optim.pt")
                self.jit_lammps(self.model)
