import torch
class Restart():
    def __init__(self):
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        
    def __call__(self,model,checkfile):
        self.forward(model,checkfile)
    
    def forward(self,model,checkfile):
        checkpoint = torch.load(checkfile,map_location=torch.device(self.device))
        model.load_state_dict(checkpoint)
