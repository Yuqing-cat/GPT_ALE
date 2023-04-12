import os
import sys
import argparse
import pandas as pd
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from active_learning.helpers import load_config

def main():
    dataset = config['data']['dataset']
    subset = config['data']["subset"]
    
    if args.suffix == "_proc":
        groupby = "label"
    else:
        groupby = 'target'

    if subset is None:
        data_path = os.path.join(os.path.join(config['data']['data_path'], dataset))
    else:
        data_path = os.path.join(os.path.join(config['data']['data_path'], dataset + "_" + subset))

    for split in ["test", "train"]:
        filename = os.path.join(data_path, "data_%s%s.csv" % (split, args.suffix))
        orig_filename = os.path.join(data_path, "data_%s%s_orig.csv" % (split, args.suffix))

        if not os.path.exists(orig_filename):
            shutil.copy2(filename, orig_filename)
        else:
            print("orig file already exists. We will re-use this, instead of copying the file again.")

        df = pd.read_csv(filename)
        df = df.groupby(groupby, group_keys=False).apply(lambda x: x.sample(args.samples_per_class, replace=False))
        
        df.reset_index(drop=True, inplace=True)
        out_filename = os.path.join(data_path, "data_%s%s.csv" % (split, args.suffix))
        df.to_csv(out_filename, mode='w', header=True, index=False)


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="configs/dbpedia_14/0")
    argparser.add_argument("--samples_per_class", type=int, default=3000)
    argparser.add_argument("--suffix", type=str, default="_proc")
    args = argparser.parse_args()

    config = load_config(args.config_path)

    main()
