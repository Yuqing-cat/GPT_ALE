import numpy as np
import pandas as pd
import argparse
import json
import os
import logging
import time
from helpers import load_config, merge_args_w_config, dump_config

class SME(object):
    def __init__(self, config):
        logging.info("Initializing SME")
        self.config = config
        self.budget = int(config['sme']['budget'])

        self.label_to_target = {}
        mapping_file = os.path.join(self.config["misc"]["api_path"], "unannotated", self.config["sme"]["mapping_file"])

        if not os.path.exists(mapping_file):
            logging.info("Waiting for mapping file to be created")
            while not os.path.exists(mapping_file):
                time.sleep(5)

        with open(mapping_file, "r") as f:
            self.label_to_target.update(json.load(f))

        label_to_target_sme_init = config['sme']['label_dict']
        
        label_to_target_sme = {}
        for target, label in enumerate(label_to_target_sme_init.keys()):
            label_to_target_sme[label] = target
                
        self.label_to_target.update(label_to_target_sme)

        self.target_to_label = {v: k for k, v in self.label_to_target.items()}
        if "Other" in self.label_to_target.keys():
            self.other = self.label_to_target["Other"]
        else:
            self.other = max(self.target_to_label) + 1

        logging.info("SME: \n%s\n%s\nOther: %s\n" % (self.label_to_target, self.target_to_label, self.other))

    def permute_anns(self, gt):
        percentage = self.config['sme']['error_rate']

        mask = [x > -1 for x in gt]
        subset = [x for x, m in zip(gt, mask) if m]
        permuted_indices = np.random.permutation(len(subset))
        permuted_indices = permuted_indices[:int(len(subset) * percentage)]
        for i, j in zip(permuted_indices, np.random.permutation(subset)):
            gt[mask.index(True, i)] = j

    def review(self, df):

        idxs = np.argsort(df['score'].values)[::-1]
        np.random.shuffle(idxs)
        
        # if the dataframe does not contain the column "target", we raise an error
        if 'target' not in df.columns:
            raise ValueError("The dataframe does not contain the column 'target'. We can only simulate SME if we have the ground truth.")

        gt = df["target"].values[idxs].astype(int)

        gt_new = []
        for i, g in zip(idxs, gt):
            if g in self.target_to_label:
                gt_new.append(g)
            else:
                gt_new.append(self.other)

        if self.config['sme']['error_rate'] > 0:
            self.permute_anns(gt_new)

        if self.budget is None:
            budget = min(df.shape[0], len(gt_new))
        else:
            budget = min(self.budget, df.shape[0], len(gt_new))

        for i, idx in enumerate(idxs[:budget]):
            df['ann'].values[idx] = gt_new[i]
            if gt_new[i] == self.other:
                df['label'].values[idx] = "Other"
            else:
                df['label'].values[idx] = self.target_to_label[gt_new[i]]

    def run(self, sleep_duration=3):
        unannotated_filename = os.path.join(self.config["misc"]["api_path"], "unannotated", "data.pkl")
        annotated_filename = os.path.join(self.config["misc"]["api_path"], "annotated", "new", "data.pkl")
        waiting_for_annotations_file = os.path.join(self.config["misc"]["api_path"], "waiting_for_annotations.txt")


        while True:
            logging.info("Waiting for annotation request")
            while not os.path.exists(unannotated_filename) or not os.path.exists(waiting_for_annotations_file):
                time.sleep(sleep_duration)
            logging.info("reviewing annotations")
            for i in range(3):
                try:
                    df = pd.read_pickle(unannotated_filename)
                    break
                except Exception as e:
                    logging.warn(e)
                    logging.warn("retrying in %d seconds."  % 5)
                    time.sleep(5)
            self.review(df)
            os.remove(unannotated_filename)
            os.makedirs(os.path.dirname(annotated_filename), exist_ok=True)
            df.to_pickle(annotated_filename)


def main(config):

    sme = SME(config)

    sme.run()

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config_path", type=str, default="configs/dbpedia_14/2")
    argument_parser.add_argument("--config", nargs="+", help="key-value pair for configuration settings")
    args = argument_parser.parse_args()


    config = load_config(args.config_path)
    config = merge_args_w_config(config, args)
    # dump config
    dump_config(config, suffix="sme")

    if os.path.exists(config['misc']['output_path']) == False:
        os.makedirs(config['misc']['output_path'])
    log_filename = os.path.join(config["misc"]["output_path"], 'sme.log')

    # log_filename = 'run.log'
    if config['misc']['log_level'] == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(filename=log_filename, level=log_level, filemode='w')

    main(config)
