import bitsandbytes as bnb
import torch
code = bnb.functional.create_dynamic_map()
code = code.cuda()
def DOBI_quantize(Matrix, k, code = None):
    U, S, V = torch.svd(Matrix.float())
    
    if code is None:
        code = bnb.functional.create_dynamic_map().to(U.device)
        
    V_T = V.T
    
    us_absmax, us_quan = [], []
    vt_absmax, vt_quan = [], []

    if U.shape[0] > V.shape[0]:
        limit_posi = V.shape[0]
        us_orig_part = [] 
        
        for i in range(k):
            s = S[i]
            
            quan_u, (absmax, _) = bnb.functional.quantize(U[:limit_posi, i] * s, code=code)
            us_absmax.append(absmax)
            us_quan.append(quan_u)
            us_orig_part.append(U[limit_posi:, i] * s)
        
            quan_vt, (absmax, _) = bnb.functional.quantize(V_T[i], code=code)
            vt_absmax.append(absmax)
            vt_quan.append(quan_vt)
        us_orig_part = torch.stack(us_orig_part, dim=0)
        us_orig_part = us_orig_part.T
        us_orig_part = us_orig_part.to(torch.float16)
    elif U.shape[0] < V.shape[0]:
        limit_posi = U.shape[0]
        vt_orig_part = [] 
        for i in range(k):
            s = S[i]
            
            quan_u, (absmax, _) = bnb.functional.quantize(U[:, i], code=code)
            us_absmax.append(absmax)
            us_quan.append(quan_u)
        
            quan_vt, (absmax, _) = bnb.functional.quantize(V_T[i, :limit_posi] * s, code=code)
            vt_absmax.append(absmax)
            vt_quan.append(quan_vt)
            vt_orig_part.append(V_T[i, limit_posi:] * s)
        vt_orig_part = torch.stack(vt_orig_part, dim=0)
        vt_orig_part = vt_orig_part.to(torch.float16)
    else:
        for i in range(k):
            s = S[i]
            
            quan_us, (absmax, _) = bnb.functional.quantize(U[:, i] * s, code=code)
            us_absmax.append(absmax)
            us_quan.append(quan_us)
        
            quan_vt, (absmax, _) = bnb.functional.quantize(V_T[i], code=code)
            vt_absmax.append(absmax)
            vt_quan.append(quan_vt)
        
    us_absmax = torch.stack(us_absmax, dim = 0)
    us_quan = torch.stack(us_quan, dim = 0)
    vt_absmax = torch.stack(vt_absmax, dim = 0)
    vt_quan = torch.stack(vt_quan, dim = 0)
        
    us_absmax = us_absmax.to(torch.float16)
    vt_absmax = vt_absmax.to(torch.float16)

    if U.shape[0] > V.shape[0]:
        tuple_info = ("us", us_orig_part)
    elif U.shape[0] < V.shape[0]:
        tuple_info = ("vt", vt_orig_part)
    else:
        tuple_info = None
    return us_quan, vt_quan, us_absmax, vt_absmax, tuple_info



def DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code = None):
    if code is None:
        code = bnb.functional.create_dynamic_map().to("cuda")
    # dequan
    us_absmax = us_absmax.to(torch.float32).to("cuda")
    vt_absmax = vt_absmax.to(torch.float32).to("cuda")
    us_quan =us_quan.to("cuda")
    vt_quan =vt_quan.to("cuda")
    dequan_us = bnb.functional.dequantize_no_absmax(us_quan, code = code)
    dequan_us = dequan_us.T * us_absmax
    dequan_vt = bnb.functional.dequantize_no_absmax(vt_quan, code = code)
    dequan_vt = torch.diag(vt_absmax) @ dequan_vt

    if tuple_info is not None:
        if tuple_info[0] == "us":
            dequan_us = torch.cat((dequan_us, tuple_info[1].to("cuda")), dim=0)
        elif tuple_info[0] == "vt":
            dequan_vt = torch.cat((dequan_vt, tuple_info[1].to("cuda")), dim=1)
    dequan_us = dequan_us.to(torch.float16)  
    dequan_vt = dequan_vt.to(torch.float16) 
    return dequan_us, dequan_vt