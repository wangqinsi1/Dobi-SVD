import logging
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from modules.remapping import DOBI_dequantize
from modules.module import *
def load_remapping_model(updated_model_path):
    logging.getLogger("transformers").setLevel(logging.ERROR)
        
    model_id = updated_model_path
    config = AutoConfig.from_pretrained(f"{model_id}/config.json")
    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(f"{model_id}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.float16) 
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    mapping_info = torch.load(f"{model_id}/remapping_weight.pt", map_location="cpu")
    for name, module in tqdm(model.named_modules(), desc="Dequantize the model after remaping."):
        if isinstance(module, nn.Linear) and all(x not in name for x in ['lm_head']):
            us_quan = mapping_info[name]["us_quan"]
            vt_quan = mapping_info[name]["vt_quan"]
            us_absmax = mapping_info[name]["us_absmax"]
            vt_absmax = mapping_info[name]["vt_absmax"]
            tuple_info = mapping_info[name]["tuple_info"]
            dequan_us, dequan_vt = DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code = None)

            compress_size = dequan_vt.size(0)* dequan_vt.size(1) + dequan_us.size(0)*dequan_us.size(1)
            ori_size = module.in_features * module.out_features
            if ori_size> compress_size:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                NewLayer = SVDTransformLayer_remapping(weight1 = dequan_vt.T, weight2 = dequan_us.T,
                                             bias = module.bias, name = name, device = "cuda")
                setattr(parent, attr_name, NewLayer)
                del module
            else:
                new_weight = dequan_us @ dequan_vt
                module.weight.data = new_weight.detach()
                
            mapping_info[name] = {}
            
    return model, tokenizer 



def load_unremapping_model(model_id):
    pruned_dict = torch.load(f"{model_id}/DobiSVD_Model.pt") #, map_location='cuda'
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer