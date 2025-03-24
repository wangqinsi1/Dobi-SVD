import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributions as dist
from transformers import Trainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
import numpy as np
import math
import random
from tqdm import tqdm
from pathlib import Path
import gc
import json    
from datetime import datetime
import logging

from utils.datautils import prepare_train_loaders
from modules.module import *
from modules.remapping import DOBI_dequantize
from modelutils import load_remapping_model, load_unremapping_model

def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in tqdm(range(nsamples), desc="evaluate"):
        if i == limit:
            break
         
        with torch.no_grad():
            input_ids = dataset[i:i+1,:-1].to(model.device)
            labels = dataset[i:i+1,1:].contiguous()
            # print(input_ids)
            logits = model(input_ids=input_ids, use_cache=False)[0]
            if torch.isfinite(logits).all():
                shift_logits = logits[:, :-1, :].contiguous()
                # shift_labels = labels.to(model.device)
                shift_labels = input_ids[:,1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                # neg_log_likelihood = loss.float() * seqlen
                # nlls.append(neg_log_likelihood)
                nlls.append(loss)
            # torch.cuda.empty_cache() 
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    # ppl = torch.exp(torch.cat(nlls,dim=-1).mean())
    torch.cuda.empty_cache() 
    return ppl.item()


def evaluate_commonsense(model, tokenizer, eval_dataset_name):
    import lm_eval
    from lm_eval import tasks
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=[eval_dataset_name]
    )
    
    return results['results']






def main(args):
    # setting random seed of numpy and torch
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU= torch.device('cpu')
    model_load_dtype = torch.float16
    computeSVD_dtype = torch.float32

    # load compressed model
    if not args.remapping:
        model, tokenizer = load_unremapping_model(args.updated_model_path)
        
    if args.remapping:
        model, tokenizer = load_remapping_model(args.updated_model_path)

    # Put compressed model on the GPU.
    model.to(DEV_GPU)
    if args.eval_metric == "ppl":
        valid_datasets = {"wikitext2", "c4", "ptb"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"eval_dataset is invalid，must be one of {valid_datasets}.")
            
        # set path
        DATASET_NAME = args.eval_dataset
        path_head_folder = Path(args.path_head_folder)
        path_head_folder_output = Path(args.path_head_folder_output)
        data_cache_dir = path_head_folder / 'data_cache' / DATASET_NAME
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        assert data_cache_dir.exists() is True
        dataset_cache_dir = path_head_folder / 'datasets' / DATASET_NAME
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        assert dataset_cache_dir.exists() is True
        # prepared dataset
        tokenized_traindata, tokenized_valdata = prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args)
        
        n_calib_samples = args.n_eval_samples
        input_ids = torch.cat([_["input_ids"].unsqueeze(0) for _ in tokenized_valdata], 0)
        ppl = evaluate_perplexity(model, input_ids, n_calib_samples)
        print(f"Perplexity on {DATASET_NAME} is: ", ppl)

    
    if args.eval_metric == "accuracy":
        valid_datasets = {"arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"eval_dataset is invalid，must be one of {valid_datasets}.")
            
        accuracy = evaluate_commonsense(model, tokenizer, args.eval_dataset)
        print(accuracy)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
#==========================================================================    
    # Settings related to the model.
    
    parser.add_argument(
        "--updated_model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )

    parser.add_argument(
        "--remapping",
        action="store_true",
        default=False,
        help="whether to use remaping to save model",
    )
    
    parser.add_argument(
        '--seq_len',
        type=int,
        default=2048,
        help='sequence length of model'
    )

    parser.add_argument(
        "--eval_metric",
        type=str,
        default="ppl",
        choices=["ppl", "accuracy"],
        help="evaluation metric",
    )
#==========================================================================    
        # Settings related to the Path.
    parser.add_argument(
        '--path_head_folder',
        type=str,
        default='./',
        help='path of the model and dataset'
    )

    parser.add_argument(
        '--path_head_folder_output',
        type=str,
        default='./results',
        help='path of the output result'
    )
    
#==========================================================================      
    # Settings related to the device.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
#==========================================================================
    # Settings related to the datasets
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=256,
        help="number of samples used for finetuning",
    )
    
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=256,
        help="number of samples used for evaluation",
    )
    
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"],
        help="finetuning dataset",
    )

    parser.add_argument(
        "--SAVE",
        action="store_true",
        help="whether to save the generated dataset",
    )

    parser.add_argument(
        "--RECREATE",
        action="store_true",
        help="whether to regenerate the dataset",
    )

 
    
 
    args = parser.parse_args()

    main(args)