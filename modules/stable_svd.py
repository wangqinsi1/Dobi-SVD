import torch.nn as nn
import torch
computeSVD_dtype = torch.float32
model_load_dtype = torch.float16
val_epsilon = 1e-10  
grad_epsilon = 1e2
K = 10
diff_epsilon = 1e-10  
nTaylor = K + 1


class stable_lowrank_SVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank):
        # print('forward: Custom_lowrank_SVD')
        U, S, V = torch.svd_lowrank(x, q=rank, niter=2)
        ctx.save_for_backward(U, S, V)
        return U, S, V
    
    @staticmethod
    def backward(ctx, gU, gS, gV):

        U, S, V = ctx.saved_tensors
        Vh = V.T
        gVh = gV.T
        
        assert U.dim() >= 2 and Vh.dim() >= 2

        if gS is None and gU is None and gVh is None:
            return None

        if gU is None and gVh is None:
            return U @ torch.diag_embed(gS) @ Vh

        '''taylor and set epsilon algorithm
        '''
        I = torch.eye(S.shape[0], device=S.device).to(S.device)
        mask_no_I = ~I.bool()
        mask_lower_tri = torch.tril(torch.ones(S.size(0), S.size(0), dtype=torch.bool)).to(S.device) # 只对左下对角线进行运算
        mask_init = mask_no_I & mask_lower_tri
        
        S_clamp = torch.clamp(S, min=val_epsilon)
        lambda_i = S_clamp.unsqueeze(0)
        lambda_j = S_clamp.unsqueeze(1)
        ratio  = lambda_j / lambda_i
        
        # form E
        E = torch.ones_like(ratio).to(S.device)
        
        # [1]too small
        mask_too_small = mask_init & (ratio == 1) & (lambda_i == val_epsilon)
        E[mask_too_small] = grad_epsilon
        
        # [2]taylor - arithmetic sequence
        mask_normal = mask_init & (~mask_too_small)
        del mask_too_small,mask_init
        diff = torch.abs(lambda_i - lambda_j)
        mask_normal_and_equal = mask_normal & (diff == 0) # arithmetic sequence
        i_mtrx = S_clamp.repeat(S_clamp.shape[0], 1) # get lambda
        E[mask_normal_and_equal] = (1/(i_mtrx[mask_normal_and_equal].pow(2))) * nTaylor
        del mask_normal_and_equal, lambda_i, lambda_j
        # [3]taylor - geometric sequence
        mask_normal_and_close = mask_normal & (diff <= diff_epsilon) & (diff > 0) # geometric sequence
        q_2 = ratio[mask_normal_and_close].pow(2)
        E[mask_normal_and_close] = (1/(i_mtrx[mask_normal_and_close].pow(2))) * ((1- q_2**nTaylor)/(1-q_2))
        del mask_normal_and_close, q_2, ratio
        # [4]other
        mask_other = mask_normal & (diff > diff_epsilon)
        j_mtrx = i_mtrx.t()
        E[mask_other] = 1 / ((i_mtrx[mask_other] - j_mtrx[mask_other])*(i_mtrx[mask_other] + j_mtrx[mask_other]))
        del mask_other, diff, i_mtrx, j_mtrx
        
        mask_pad = mask_no_I & (~mask_lower_tri)
        E[mask_pad] = -1. * (E.t()[mask_pad])
        del mask_no_I, mask_pad,mask_lower_tri
        # ic(1/E)

        skew = lambda X: X - X.T
        
        UhgU_skew = skew(U.T @ gU) * E if gU is not None else torch.zeros_like(U)
        VhgV_skew = skew(Vh @ gVh.T) * E if gVh is not None else torch.zeros_like(Vh)
        gA_core = U @ (UhgU_skew @ S.diag_embed() + S.diag_embed() @ VhgV_skew + torch.diag_embed(gS)) @ Vh

        gUSinv = gU / S_clamp.unsqueeze(-2)
        additional_term1 = (gUSinv - U@(U.T @ gUSinv))@Vh
        
        SinvgVh = gVh / S_clamp.unsqueeze(-1)
        additional_term2 = U @ (SinvgVh -(SinvgVh @ Vh.T) @ Vh)
        
        gA_core += additional_term1 +additional_term2

        return gA_core,None












class SVDTransformLayer(nn.Module):
    def __init__(self, gamma = None, SEQ_LEN = None, beta = None,
                 input_size = None, output_size = None, 
                 weight_size = None, weight = None, 
                 bias = None, name = None, device = None):
        super(SVDTransformLayer, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=computeSVD_dtype)).to(device) # gamma is trainable
        if bias is None:
            self.ori = nn.Linear(input_size, output_size, bias=False).to(device)
        else:
            self.ori = nn.Linear(input_size, output_size, bias=True).to(device)
            self.ori.bias = nn.Parameter(bias).to(device)
        self.ori.weight = nn.Parameter(weight).to(device)
        if name:
            self.name = name
        self.ori_weight_size = weight_size
        in_plus_out_size = input_size + output_size
        nblocks = input_size / SEQ_LEN
        self.nblocks_total_size = in_plus_out_size * nblocks
        self.beta = beta
    
    def forward(self, x):
        x= self.ori(x).to(computeSVD_dtype)
        # real_x= x.to(model_load_dtype)
        
        if x.dim() == 3:
            x = x.squeeze(0)
            squeeze_need = 1
        else:
            squeeze_need = 0
            
        assert x.dim() == 2
        m,n =x.shape
        full_rank = min(m,n)
        
        gamma_range = self.gamma.detach()
        gamma_range = int(gamma_range.int()+5)
        gamma_range = min(full_rank, max(1, gamma_range))
        
        U, S, V = stable_lowrank_SVD.apply(x,gamma_range)
        sequence = torch.arange(1, len(S)+1).to(x.device)
        real_gamma = min(len(S), max(1, self.gamma))
        
        Trunc = (0.5*torch.tanh(self.beta * (real_gamma - sequence))+0.5)
        S_transformed = (S) * (Trunc)
        S_diag = torch.diag_embed(S_transformed)
        x_transformed = torch.matmul(torch.matmul(U, S_diag), V.T)   #U @ S_diag @ V.T
        real_x= x_transformed.to(model_load_dtype)
        
        if squeeze_need == 1:
            real_x = real_x.unsqueeze(0)
        return real_x