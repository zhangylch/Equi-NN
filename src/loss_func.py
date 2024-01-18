def loss_func(prediction,label,weight):
    lossprop=torch.cat([torch.sum(torch.square(ipred-ilabel)).reshape(-1) for ipred, ilabel in zip(prediction,label)])
    loss=torch.inner(lossprop,weight)
    return loss,lossprop
