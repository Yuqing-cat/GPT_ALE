import os
import json
import time
import pandas as pd
import shutil
from datetime import datetime
import logging
import requests
import numpy as np
import glob
import argparse

def flatten_config(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def request_annotations(dataset, config, preds=None, confs=None):
    # this contains the data we would like to have annotated
    unannotated_filename = os.path.join(config["misc"]["api_path"], "unannotated", "data.pkl")
    # this file is used by api/ui to know current mapping of targets to labels
    mapping_filename = os.path.join(config["misc"]["api_path"], "unannotated", "mapping.json")

    os.makedirs(os.path.dirname(unannotated_filename), exist_ok=True)

    # this is so that the method works on subsets as well
    if hasattr(dataset, 'indices'):
        df = dataset.dataset.df.iloc[dataset.indices]
        dataset = dataset.dataset
    else:
        df = dataset.df

    # set label to "unknown" where annotation is -1. This is because huggingface datasets come with label
    # we don't just set all to "unknown", because we may already have some annotations
    idx = np.where(df['ann'] == -1)[0]
    df['label'].values[idx] = 'unknown'

    # we make a copy of existing anns, and set all anns to -1
    df['ann_orig'] = df['ann']
    df.loc[:, 'ann'] = -1

    # this is naughty, but a quick fix for huggingface datasets that don't have that column
    # please fix this by instead not important that column anywhere, and also update the api
    if not 'title' in df.columns:
        df['title'] = ""

    if preds is not None:
        df_out = df.copy()
        labels = [dataset.target_to_label[pred] for pred in preds]
        df_out['label'] = labels[:df.shape[0]] # TODO, don't ovrwrite the labels already provided by SME!
        df_out['confs'] = confs[:df.shape[0]]
        df_out.to_pickle(unannotated_filename)
    else:
        df.to_pickle(unannotated_filename)
    
    ltt, _, _ = dataset.update_label_target_mapping()

    with open(mapping_filename, "w") as f:
        json.dump(ltt, f, cls=NpEncoder)

    
    logging.debug("Requested annotations")


def check_for_annotations(dataset, config, wait=False, sleep_duration=1, format_str="_%Y%b%d_%H%M%S"):
    annotated_filename = os.path.join(config["misc"]["api_path"], "annotated", "new", "data.pkl")
    waiting_for_annotations_file = os.path.join(config["misc"]["api_path"], "waiting_for_annotations.txt")
        
    if not os.path.exists(annotated_filename):
        if wait == True: # we wait until the file shows up
            # send signal to SME that we are waiting for annotations
            with open(waiting_for_annotations_file, "w") as f:
                f.write(str(time.time()) + "\n")
            logging.info("Waiting for annotations")
            while not os.path.exists(annotated_filename):
                time.sleep(sleep_duration)
        else: # we just return, without new annotations
            return False
    
    logging.debug("Received annotations")

    # we read the annotations
    attempts = 10
    df_ann = None
    for attempt in range(attempts):
        try:
            df_ann = pd.read_pickle(annotated_filename)
            break
        except:
            logging.warning("Could not read annotations. Trying again in %d seconds." % (attempt + 1))
            time.sleep(attempt + 1)

    # we were unable to load new annotations. Most likely a connection issue
    if df_ann is None:
        return False

    # we signal that we are done waiting
    if wait == True and os.path.exists(waiting_for_annotations_file):
        os.remove(waiting_for_annotations_file)

    if hasattr(dataset, 'indices'):
        df = dataset.dataset.df
        target_to_label = dataset.dataset.target_to_label
    else:
        df = dataset.df
        target_to_label = dataset.target_to_label

    df_idx = df_ann.index

    # note that we are not calculating the difference between new and old annotations
    # but the two should be identical, except for changes made by the user
    # and the fact that labels in df_ann contains model predictions
    df.loc[df_idx,'ann'] = df_ann.loc[df_idx,'ann']
    # df.loc[df_idx, 'ann_by'] = 'sme'
    labels = []
    # cases for ann_by and label handling
    # the sample was already labeled by gpt3/sme
    # # gpt3: ann!=-1, ann_by="gpt3",        label!="unknown"
    # # new : ann!=-1, ann_by=[gpt3,"", sme], label!="unknown"
    # # sme : ann!=-1, ann_by"sme",           label!="unknoqn"
    # # if by gpt3. we need to make sure we don't copy the incorrect label over and pretend it came from the sme
    # # if by sme, labels can change, because sme's change their minds
    # it wasn't labeled yet
    # # this is the easy case, just accept the new label from SME
    ann_by = []
    anns = []
    mmc = 0
    for row in df_ann.itertuples():
        if row.ann == -1: # we did not receive a label from the sme for this sample
            if row.ann_orig == -1: # this sample also wasn't annotated before
                anns.append(-1)
                labels.append('unknown')
                ann_by.append("")
            else: # this sample was NOT annoated by sme this time, but had been annotated (by sme or gpt3)
                anns.append(row.ann_orig)
                labels.append(target_to_label[row.ann_orig]) # don't just copy label, maybe gt or student model guess, and we want the label corresponding to row.ann_orig (which might be wrong)
                # labels.append(row.label)
                ann_by.append(row.ann_by)
                # if target_to_label[row.ann_orig] != row.label:
                #     if row.ann_by == 'sme':
                #         print()
                #     mmc += 1
                #     print(target_to_label[row.ann_orig])
                #     print(row.label)
                #     print(row.ann_by)
                #     print(row.ann)
                #     print(row.target)
                #     print(row.ann_orig)
                #     print()
        else: # sme provided an annotation (ann) for this sample
            if row.ann in target_to_label.keys():
                labels.append(target_to_label[row.ann])
            else:
                # we received a new label/target mapping from sme
                labels.append(row.label)
            ann_by.append('sme')
            anns.append(row.ann)

    df.loc[df_idx,'label'] = labels
    df.loc[df_idx,'ann_by'] = ann_by
    df.loc[df_idx,'ann'] = anns

    # we set column "sampler" to None for all the rows where column ann_by is not sme, 
    # because we only want to track that if the sme has annotated the samples
    df['sampler'] = df['sampler'].where(df['ann_by'] == 'sme', "")

    timestamp = datetime.now().strftime(format_str)
    archive_filename = os.path.join(config["misc"]["api_path"], "annotated", "archive", "data" + timestamp + ".pkl")
    os.makedirs(os.path.dirname(archive_filename), exist_ok=True)
    shutil.move(annotated_filename, archive_filename)

    logging.info("Received new annotations.")
    # we return true if we received new annotations
    return True


def lock(config):
    lock_file = os.path.join(config["misc"]["api_path"], "lock.txt")

    with open(lock_file, "w") as f:
        f.write(str(time.time()) + "\n")


def unlock(config):
    lock_file = os.path.join(config["misc"]["api_path"], "lock.txt")

    if os.path.exists(lock_file):
        os.remove(lock_file)

def merge_recursive_dict(dict1, dict2, skip_merge_on_keys=['label_dict', 'label_subset']):
    """
    Recursively merge two dictionaries.
    """
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # If both entries are dictionaries, merge them recursively
            # unless the key is in skip_merge_on_keys
            if key in skip_merge_on_keys:
                dict1[key] = dict2[key]
            else:
                merge_recursive_dict(dict1[key], dict2[key])
        else:
            # Otherwise, simply overwrite the entry in dict1 with the entry in dict2
            dict1[key] = dict2[key]
    return dict1

def load_config_files(config_files):
    config = {}
    for cf in config_files:
        with open(cf, "r") as f:
            config.update(json.load(f))

    return config

def load_config(config_path):
    config_files = glob.glob(os.path.join(config_path, "*.json"))

    if len(config_files) == 0:
        raise Exception("No config files found in {}".format(config_path))

    config = load_config_files(config_files)

    # we load the defaults for the dataset    
    config_path_defaults = os.path.join(os.path.dirname(config_path), "defaults")
    config_files_defaults = glob.glob(os.path.join(config_path_defaults, "*.json"))

    if len(config_files_defaults) == 0:
        raise Exception("No config files found in {}".format(config_path_defaults))

    config_defaults = load_config_files(config_files_defaults)

    config = merge_recursive_dict(config_defaults, config)

    return config

def merge_args_w_config(default_config: dict, args: argparse.Namespace) -> dict:
    config = default_config.copy()
    if args.config:
        for key_value_pair in args.config:
            key, value = key_value_pair.split("=")
            keys = key.split(".")
            d = config
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            if isinstance(value, dict):
                d[keys[-1]] = json.loads(value)
            elif isinstance(value, str):
                try:
                    d[keys[-1]] = float(value)
                except:
                    d[keys[-1]] = value
            else:
                raise ValueError("type not supported: %s" % type(value))
    return config

def dump_config(config, suffix="run"):
    config_path = config["misc"]["output_path"]
    config_file = os.path.join(config_path, "settings_%s.json" % suffix)
    os.makedirs(config_path, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

def main():
    return

if __name__ == "__main__":

    main()