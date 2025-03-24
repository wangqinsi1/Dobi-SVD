import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import random
from tqdm import tqdm

def prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args):
    traindata_cache_file = dataset_cache_dir / f"traindata.pt"
    valdata_cache_file = dataset_cache_dir / f"valdata.pt"
    LOAD = traindata_cache_file.exists() and valdata_cache_file.exists()
    SEQ_LEN = args.seq_len
    NSAMPLES_train = args.n_train_samples
    NSAMPLES_val = args.n_eval_samples
    SAVE = args.SAVE
    SEED = args.seed
    
    if LOAD and not args.RECREATE:
        traindata = torch.load(traindata_cache_file)
        valdata = torch.load(valdata_cache_file)
    else:
        if DATASET_NAME == 'c4':
            traindata = load_dataset("json", 
                                      data_files={"train": str(dataset_cache_dir / "en/c4-train.00000-of-01024.json.gz")},
                                      cache_dir=dataset_cache_dir,
                                      split="train")
            valdata = load_dataset("json", 
                                    data_files={"validation": str(dataset_cache_dir /"en/c4-validation.00000-of-00008.json.gz")},
                                    cache_dir=dataset_cache_dir,
                                    split="validation")
        elif DATASET_NAME == 'wikitext2':
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
            valdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)
        elif DATASET_NAME == 'ptb':
            traindata = load_dataset("ptb_text_only", "penn_treebank", split="train", cache_dir=dataset_cache_dir)
            valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation", cache_dir=dataset_cache_dir)
        if SAVE:
            torch.save(traindata, traindata_cache_file)
            torch.save(valdata, valdata_cache_file)
            print("Training and validation data has been processed and saved!")
    
    print("Training and validation has been loaded!")
    
    
    tokenized_traindata_cache_file = data_cache_dir / f"traindata_{DATASET_NAME}_{NSAMPLES_train}_{SEQ_LEN}.pt"
    tokenized_valdata_cache_file = data_cache_dir / f"valdata_{DATASET_NAME}_{NSAMPLES_val}_{SEQ_LEN}.pt"
    LOAD = tokenized_traindata_cache_file.exists() and tokenized_valdata_cache_file.exists()
    
    if LOAD and not args.RECREATE:
        tokenized_traindata = torch.load(tokenized_traindata_cache_file)
        tokenized_valdata = torch.load(tokenized_valdata_cache_file)
    else:
        # traindata
        tokenized_traindata = []
        tokenized_valdata = []
        if DATASET_NAME == 'c4':
            random.seed(SEED)
            for _ in tqdm(range(NSAMPLES_train), desc="Processing training data", unit=" sample"):
                while True:
                    i = random.randint(0, len(traindata) - 1)
                    trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                    if trainenc.input_ids.shape[1] >= SEQ_LEN:
                        break
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_traindata.append({"input_ids": inp, "attention_mask": attention_mask})
                
            # val
            random.seed(0)
            for _ in tqdm(range(NSAMPLES_val), desc="Processing validate data", unit=" sample"):
                while True:
                    i = random.randint(0, len(valdata) - 1)
                    tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                    if tmp.input_ids.shape[1] >= SEQ_LEN:
                        break
                if tmp.input_ids.shape[1] - SEQ_LEN - 1 <= 0:
                    i = 0
                else:
                    i = random.randint(0, tmp.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = tmp.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_valdata.append({"input_ids": inp, "attention_mask": attention_mask})

        
        elif DATASET_NAME == 'wikitext2' or DATASET_NAME == 'ptb':
            if DATASET_NAME == 'wikitext2':
                train_tot_text = "\n\n".join(traindata["text"])
                val_tot_text = "\n\n".join(valdata["text"])
            elif DATASET_NAME == 'ptb':
                train_tot_text = "\n\n".join(traindata["sentence"])
                val_tot_text = "\n\n".join(valdata["sentence"])
            # train
            random.seed(SEED)
            for s in tqdm(range(NSAMPLES_train), desc="Processing training data", unit=" sample"):
                i = random.randint(0, len(train_tot_text) - SEQ_LEN - 1)
                j = i + SEQ_LEN * 10
                trainenc = tokenizer(train_tot_text[i:j], return_tensors="pt")
                if trainenc.input_ids.shape[1] < SEQ_LEN:
                    s = s - 1
                    continue
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_traindata.append({"input_ids": inp, "attention_mask": attention_mask})
            
            # val
            random.seed(0)
            for s in tqdm(range(NSAMPLES_val), desc="Processing validation data", unit=" sample"):
                i = random.randint(0, len(val_tot_text) - SEQ_LEN - 1)
                j = i + SEQ_LEN * 10
                trainenc = tokenizer(val_tot_text[i:j], return_tensors="pt")
                if trainenc.input_ids.shape[1] < SEQ_LEN:
                    s = s - 1
                    continue
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_valdata.append({"input_ids": inp, "attention_mask": attention_mask})
            
        if SAVE:
            torch.save(tokenized_traindata, tokenized_traindata_cache_file)
            torch.save(tokenized_valdata, tokenized_valdata_cache_file)
            print("Tokenized data has been processed and saved!")
    
    print("Tokenized data has been loaded!")
    
    return  tokenized_traindata, tokenized_valdata