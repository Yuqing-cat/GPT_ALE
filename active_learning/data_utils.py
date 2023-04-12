import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import mlflow 

class PandasDataset(Dataset):
    def __init__(self, df, config, label_to_target=None, target_to_label=None):
        self.config = config
        self.df = df
        self.selected_columns = ["embedding", "ann"]
        self.embedding_size = len(json.loads(df.iloc[0]["embedding"]))

        if label_to_target is None:
            self.label_to_target, self.target_to_label = self.get_label_target_mapping()
        else:
            self.label_to_target = label_to_target
            self.target_to_label = target_to_label

        self.n_classes = max(self.target_to_label.keys()) + 1 # +1 because we start counting at 0
        
    def get_label_target_mapping(self):
        """ Returns a mapping from label to target and vice versa."""
        label_to_target = {}
        target_to_label = {}

        anns = self.df['ann'].unique()
        for ann in anns:
            if ann == -1:
                continue
            df_s = self.df[(self.df["ann"] == ann) & (self.df["ann_by"] == "sme")].head(1)
            label = df_s['label'].values[0]
            label_to_target[label] = ann
            target_to_label[ann] = label

        return label_to_target, target_to_label
    
    def update_label_target_mapping(self):
        """ Returns a mapping from label to target and vice versa. If new labels are found, they are added to the mapping."""
        ltt, ttl = self.get_label_target_mapping()
        
        label_to_target = dict(self.label_to_target)

        modified = False
        for key in label_to_target.keys():
            if key not in ltt:
                modified = True
                ltt[key] = label_to_target[key]
                ttl[label_to_target[key]] = key
        
        # # not sure why this has to happen
        # for t in range(max(ttl.keys())):
        #     if t not in ttl.keys():
        #         ttl[t] = 'unknown'

        return ltt, ttl, modified

    def set_label_target_mapping(self, label_to_target, target_to_label):
        """ Sets the mapping from label to target and vice versa."""
        self.label_to_target = label_to_target
        self.target_to_label = target_to_label
        self.n_classes = max(self.target_to_label.keys()) + 1

    def __getitem__(self, idx):
        sample = []
        for key in self.selected_columns:
            value = self.df.iloc[idx][key]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                    value = np.array(value, dtype=np.float32)
                except json.decoder.JSONDecodeError:
                    pass
            sample.append(value)

        return sample

    def __len__(self):
        return len(self.df)
    
    def update_df(self, dict, idx=None):
        if idx is None:
            for key in dict.keys():
                self.df[key] = dict[key]
        else:
            for key in dict.keys():
                if dict[key] is None:
                    continue
                if isinstance(dict[key][0], list):
                    for i, ii in enumerate(idx):
                        self.df.loc[ii, key] = json.dumps(dict[key][i])
                elif isinstance(dict[key][0], np.ndarray):
                    for i, ii in enumerate(idx):
                        self.df.loc[ii, key] = json.dumps(dict[key][i].tolist())
                else:
                    self.df.loc[idx, key] = dict[key]


        
    def update_emb(self, featurizer):
        self.embeddings = torch.zeros((self.n_classes, self.embedding_size))

        for target, label in self.target_to_label.items():
            if target > -1:
                self.embeddings[target,:] = featurizer.featurize(label)

    def eval_annotations(self, ann_by='sme'):
        df = self.df
        df_ann = df[(df['ann'] != -1) & (df['ann_by'] == ann_by)]

        if df_ann.shape[0] == 0:
            return 0

        anns = df_ann['ann'].values.astype(int)
        labels = df_ann['target'].values.astype(int)

        acc = (anns == labels).sum()/len(anns)

        return acc

    def save(self, path='results', suffix="", idx=None):
        os.makedirs(path, exist_ok=True)

        if idx is None:
            self.df.to_pickle(f"{path}/df{suffix}.pkl")
        else:
            df_s = self.df.loc[idx]
            df_s.to_pickle(f"{path}/df{suffix}.pkl")

    def get_ann_counts(self, per_class=True, ann_by='sme'):
        # get the number of annotations. if per_class is True, the number of annotations per class is also returned
        df = self.df
        df_ann = df[(df['ann'] != -1) & (df['ann_by'] == ann_by)]

        ann_counts = {'total': len(df_ann)}

        if per_class:
            anns = df_ann['ann'].values.astype(int)
            for target, label in self.target_to_label.items():
                ann_counts[label] = int((anns == target).sum())
        
        return ann_counts

    def get_class_weights(self):
        ann_counts = self.get_ann_counts(per_class=True)
        class_weights = []
        for c in range(self.n_classes):
            if c in self.target_to_label.keys():
                class_weights.append((ann_counts['total'] - ann_counts[self.target_to_label[c]]) / ann_counts['total'])
            else:
                class_weights.append(0) # doesn't really matter whether this is 0 or 1, because the weights should never be called upon
        return class_weights

    def log_anns(self, step, split='val', per_class=True, destination='mlflow'):

        if split == 'val':
            ann_by = 'sme'
        else:
            ann_by = 'gpt3'

        ann_counts = self.get_ann_counts(per_class=per_class, ann_by=ann_by)

        if destination == 'mlflow':
            for key in ann_counts.keys():
                mlflow.log_metric(f"anns/{split}/{key}", ann_counts[key], step=step)
        elif destination == 'api':
            filename = os.path.join(self.config['misc']['api_path'], "annotations.json")
            with open(filename, 'w') as f:
                json.dump(ann_counts, f)
        else:
            raise ValueError(f"destination {destination} not recognized")


    def log_samplers(self, step, split='val'):
        if split == 'val':
            ann_by = 'sme'
        else:
            ann_by = 'gpt3'

        df = self.df
        df_ann = df[(df['ann'] != -1) & (df['ann_by'] == ann_by)]

        v, c = np.unique(df_ann['sampler'], return_counts=True)

        for value, count in zip(v,c):
            mlflow.log_metric(f"sampling/{split}/{value}", count, step)