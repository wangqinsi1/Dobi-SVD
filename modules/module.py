import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SVDTransformLayer(nn.Module):
    def __init__(self,gamma=None, weight = None, bias = None, name = None, device = None):
        super(SVDTransformLayer, self).__init__()
        
        if name:
            self.name = name
        
        W_T = weight.T.to(torch.float32)
        U, S, V = torch.svd_lowrank(W_T, q=int(gamma), niter = 10)
        diag_S = torch.diag(S)
        sqrt_S = torch.sqrt(diag_S)
        A_weight_T = (U @ sqrt_S).to(torch.float16)
        B_weight_T = (sqrt_S @V.T).to(torch.float16)
        if bias is None: 
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device = device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=False, device = device)
        else:
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device = device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=True, device = device)
            self.ALinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x




class SVDTransformLayer_remapping(nn.Module):
    def __init__(self, weight1 = None, weight2 = None, bias = None, name = None, device = None):
        super(SVDTransformLayer_remapping, self).__init__()
        
        if name:
            self.name = name
        
        A_weight_T = (weight1).to(torch.float16)
        B_weight_T = (weight2).to(torch.float16)
        if bias is None: 
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device = device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=False, device = device)
        else:
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device = device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=True, device = device)
            self.ALinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x