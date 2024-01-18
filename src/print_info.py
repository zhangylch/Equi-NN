import time

class Print_Info():
    def __init__(self,end_lr):
        self.end_lr=end_lr
        self.ferr=open("nn.err","w")                    
        self.ferr.write("Equivariant MPNN package based on three-body descriptors \n")
        self.ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

        

    def __call__(self,iepoch,lr,loss_train,loss_val):
        self.forward(iepoch,lr,loss_train,loss_val) 
   
    def forward(self,iepoch,lr,loss_train,loss_val):
        #output the error 
        self.ferr.write("Epoch= {:6},  lr= {:5e}  ".format(iepoch,lr))
        self.ferr.write("train: ")
        for error in loss_train:
            self.ferr.write("{:10e} ".format(error))
        self.ferr.write(" validation: ")
        for error in loss_val:
            self.ferr.write("{:10e} ".format(error))
        self.ferr.write(" \n")
        self.ferr.flush()
        if lr<=self.end_lr: 
            self.ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
            self.ferr.close()
        
