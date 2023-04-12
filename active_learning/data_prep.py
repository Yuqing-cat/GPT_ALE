import glob
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import json
import pandas as pd
import os
import argparse
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from active_learning.helpers import load_config

def adjust_class_imbalance(df, split, config):
    # if we have not defined a subset in config.json, we return
    if len(config['data']["label_subset"]) == 0:
        return

    # We drop the samples that are not in the subset
    # For those that are in the subset, we only keep the desired fraction
    uq_labels = df['label'].unique()
    drop_indices = []
    for label in uq_labels:
        idx = list(df.index[df['label'] == label])
        if label in config['data']["label_subset"].keys():
            idx = np.random.choice(idx, int((1 - config['data']["label_subset"][label][split]) * len(idx)), replace=False)
        drop_indices.extend(idx)

    df.drop(drop_indices, inplace=True)

    df.reset_index(inplace=True, drop=True)

    return

def align_w_sme(df, config):
    # We align the targets and labels with the SME
    
    # if there is no target column, we return
    # (there is no way of knowing which samples to remove)
    if 'target' not in df.columns:
        return

    # if the SME is not defined in sme.json, we return
    if 'sme' not in config or 'label_dict' not in config['sme']:
        return

    label_to_target = config['sme']['label_dict']
    target_to_label = {v: k for k, v in label_to_target.items()}

    # dictionary of indices for each target
    idx = {}
    for label in label_to_target.keys():
        if label == 'unknown':
            continue
        idx[label] = list(df.index[df['target'] == label_to_target[label]])
    idx["Other"] = list(df.index[(df['target'].isin(target_to_label.keys()) == False)])

    for target, label in enumerate(idx.keys()):
         df['target'].values[idx[label]] = target
         df['label'].values[idx[label]] = label

def generate_embedding(samples, model, tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokens_batch = tokenizer.batch_encode_plus(samples, truncation=True, padding=True, return_tensors="pt")
    for key in tokens_batch.keys():
        tokens_batch[key] = tokens_batch[key].to(device)
    with torch.no_grad():
        outputs = model(**tokens_batch)
    
    outputs = outputs.logits.cpu().numpy().tolist()

    json_out = []
    for i in range(len(samples)):
        json_out.append(
            json.dumps(outputs[i])
        )

    return json_out

def save_results(df, embeddings, input_filename, current_row):
    r = len(embeddings)

    df_s = df.iloc[(current_row - r + 1):(current_row + 1)].copy()
    df_s.loc[:, 'embedding'] = embeddings

    output_filename = input_filename.replace(".csv", "_proc.csv")

    new_file = True if r == (current_row + 1) else False
    if new_file and os.path.exists(output_filename):
        os.remove(output_filename)
    df_s.to_csv(output_filename, mode='a', header=new_file, index=False)


def main():
    dataset = config['data']['dataset']
    subset = config['data']["subset"]

    add_column_dict= {'corr' : -1, 'probs' : None, 'score' : 0, 'ann' : -1, 'embedding' : None}

    if subset is None:
        data_path = os.path.join(os.path.join(config['data']['data_path'], dataset))
    else:
        data_path = os.path.join(os.path.join(config['data']['data_path'], dataset + "_" + subset))
        
    if os.path.exists(data_path):
        filenames = glob.glob(data_path + "/*.csv")

    filenames = [f for f in filenames if "_proc" not in f]

    if len(filenames) == 0:
        raise Exception("No files found in %s" % data_path)

    filenames.sort()

    model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'])
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Identity(in_features)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint'])
    
    for filename in filenames:
        print("filename:", filename)
        df = pd.read_csv(filename, header=0, sep='\t')
        for k, v in add_column_dict.items():
                    df[k] = v

        embeddings = []
        samples = []
        for r, row in tqdm(enumerate(df.itertuples()), total=len(df)):
            samples.append(row.text)
            if (r + 1) % args.batch_size == 0:
                embeddings_cur = generate_embedding(samples, model, tokenizer)
                embeddings.extend(embeddings_cur)
                samples = []

            if (r + 1) % (100 * args.batch_size) == 0:
                print("saving progress ...")
                save_results(df, embeddings, filename, r)
                embeddings = []

        if len(samples) > 0:
            embeddings_cur = generate_embedding(samples, model, tokenizer)
            embeddings.extend(embeddings_cur)

            save_results(df, embeddings, filename, r)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="configs/ag_news/0")
    argparser.add_argument("--batch_size", type=int, default=8, help="Batch size for tokenizer and model. (Default: 8) Suprisingly, larger batch sizes have not been found to improve speed")
    args = argparser.parse_args()

    config = load_config(args.config_path)

    main()
