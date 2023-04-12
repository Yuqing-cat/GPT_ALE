from datasets import load_dataset
import pandas as pd
import csv
import os
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from active_learning.helpers import load_config

def get_target_to_label(label_to_target):
    target_to_label = {}
    for label, target in label_to_target.items():
        target_to_label[target] = label
    return target_to_label

def main(config):
    
    dataset = config['data']['dataset'] # name of huggingface dataset
    subset = config['data']["subset"] # name of subset of dataset
    text_col = config['data']['text_column'] # name of text column (input)
    label_col = config['data']['label_column'] # name of label column (target)
    label_to_target = config['data']['label_dict']  # this is missing from config
    target_to_label = get_target_to_label(label_to_target)

    raw_datasets = load_dataset(dataset, subset)

    for split in list(raw_datasets.keys()):
        df = pd.DataFrame(raw_datasets[split])

        df.rename(columns={label_col:'target', text_col: 'text'}, inplace=True)

        df['label'] = df['target'].map(target_to_label)
        df = df[['text', 'label', 'target']]

        if subset is None:
            out_path = os.path.join(os.path.join("data/", dataset))
        else:
            out_path = os.path.join(os.path.join("data/", dataset + "_" + subset))

        os.makedirs(out_path, exist_ok=True)

        out_filename = os.path.join(out_path, "data_%s.csv" % split)

        df.to_csv(out_filename, index=False, quoting=csv.QUOTE_NONNUMERIC, sep="\t")
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="configs/ag_news/0")
    args = argparser.parse_args()

    config = load_config(args.config_path)

    main(config)
