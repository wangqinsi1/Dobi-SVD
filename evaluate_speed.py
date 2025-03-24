import argparse
import os
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
import torch
import numpy as np
import random
import itertools
import torch
import time
import random
import numpy as np
 

from modelutils import load_remapping_model, load_unremapping_model

def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader




def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):

    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)
    weight_memory = torch.cuda.memory_allocated()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()

        try:
            generation_output = model.generate(
                    input_ids=batch,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    use_cache=True,
                    top_k=50,
                    max_length=original_len + generated_len,
                    top_p=0.95,
                    temperature=1,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)

            if torch.isfinite(generation_output).all():
                throughput += (end_time - start_time)
                print(f"Batch {batch_idx+1}: Time {end_time - start_time:.2f} sec")

        except RuntimeError as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()  # 避免内存不足
            token_num -= batch.shape[0] * generated_len  # 移除无效的token数量

    total_memory_gb = end_memory / (1024 ** 3)
    weight_memory_gb = weight_memory / (1024 ** 3)
    activation_memory_gb = (end_memory - start_memory) / (1024 ** 3)

    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Weight Memory: {weight_memory_gb:.2f} GB")
    print(f"Activation Memory: {activation_memory_gb:.2f} GB")
    
    if throughput > 0:
        print(f"Throughput: {token_num / throughput:.2f} tokens/sec")
    else:
        print("Throughput could not be calculated due to errors.")






def main(args):
    # setting random seed of numpy and torch
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU = torch.device('cpu')
    
    # load compressed model
    if not args.remapping:
        model, tokenizer = load_unremapping_model(args.updated_model_path)
        
    if args.remapping:
        model, tokenizer = load_remapping_model(args.updated_model_path)

    # Put compressed model on the GPU.
    model.to(DEV_GPU)
    
    eff_eval(model, tokenizer, dataset='wikitext2', generated_len=args.generated_len, batch_size=args.batch_size, device=DEV_GPU)





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
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="finetuning dataset",
    )

    
    parser.add_argument(
        "--generated_len",
        type=int,
        default=64,
        help="length of generated tokens",
    )
    
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for evaluation",
    )
    
    

  
 
    args = parser.parse_args()

    main(args)
