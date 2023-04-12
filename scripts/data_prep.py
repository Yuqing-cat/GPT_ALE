from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import json
import pandas as pd
import os
import argparse
from sqlalchemy import create_engine
import urllib
from tqdm import tqdm

def load_config(config_files):
    config = {}
    for config_file in config_files:
        with open(config_file, "r") as f:
            config.update(json.load(f))
    return config

def add_clustered_index(table, engine):
    print("adding clustered index")

    stmt = "DROP INDEX IF EXISTS %s_idx ON %s" % (table, table)
    _ = engine.execute(stmt)

    # primary index as to be NOT NULL
    stmt = "ALTER TABLE %s alter column idx bigint NOT NULL" % table
    _ = engine.execute(stmt)

    # add primary key
    stmt = """ALTER TABLE %s
            ADD CONSTRAINT %s_idx PRIMARY KEY CLUSTERED (idx)""" % (table, table)
    _ = engine.execute(stmt)

def test_table(table, engine):
    print("Testing connection to SQL server.")
    stmt = "SELECT * FROM %s" % table
    res = engine.execute(stmt)
    row = res.fetchone()
    print(row)

def create_sql_engine(config):
    conn = f"""Driver={config['sql']['driver']};Server=tcp:{config['sql']['server']},1433;Database={config['sql']['database']};
    Uid={config['sql']['username']};Pwd={config['sql']['password']};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""

    params = urllib.parse.quote_plus(conn)
    conn_str = 'mssql+pyodbc:///?autocommit=true&odbc_connect={}'.format(params)
    engine = create_engine(conn_str,echo=False,fast_executemany=True, pool_size=1000, max_overflow=100)

    print('connection is ok')

    return engine

def generate_embedding(text, model, tokenizer, device='cuda'):
    tokens = tokenizer(text, truncation=True)
    for key in tokens.keys():
        tokens[key] = torch.tensor(tokens[key]).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.logits.squeeze(0).cpu().numpy().tolist()

def main(args):
    config = load_config([args.config, args.sql_config])

    dataset = config['data']['dataset']
    subset = config['data']["subset"]
    
    if subset is None:
        out_path = os.path.join(os.path.join("active_learning/data/", dataset))
    else:
        out_path = os.path.join(os.path.join("active_learning/data/", dataset + "_" + subset))

    df_filename = os.path.join(out_path, "data.csv")
    json_filename = os.path.join(out_path, "label_dict.json")

    df = pd.read_csv(df_filename, sep="\t")
    with open(json_filename, "r") as f:
        label_to_target = json.load(f)

    dict_df = pd.DataFrame(
        {'label': list(label_to_target.keys()), 'target': list(label_to_target.values())}
    )

    add_column_dict= {'corr' : -1, 'probs' : None, 'embedding' : None, 'score' : 0, 'ann' : -1, 'version_tracking' : None, 'row_number' : None, 'label' : None, "idx" : 0}
    df = pd.concat([df, pd.DataFrame(add_column_dict, index=df.index)], axis=1)

    model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'])
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Identity(in_features)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint'])

    for r in tqdm(range(df.shape[0])):
        embedding = generate_embedding(df['text'][r], model, tokenizer, device)
        df.loc[r, 'embedding'] = json.dumps(embedding)
        df.loc[r, 'idx'] = r

    engine = create_sql_engine(config)

    table_name = config['data']['dataset']
    try:
        print("creating table")
        print("table name:", table_name)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, method='multi', chunksize=100)
    except Exception as e:
        print(e)
        print("failed")

    add_clustered_index(table_name, engine)

    test_table(table_name, engine)

    table_name = config['data']['dataset'] + "_dict"
    try:
        print("creating table")
        print("table name:", table_name)
        dict_df.to_sql(table_name, con=engine, if_exists='replace', index=False, method='multi', chunksize=100)
    except Exception as e:
        print(e)
        print("failed")


    engine.dispose()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", type=str, default="config.json")
    argument_parser.add_argument("--sql_config", type=str, default="config_sql.json")

    args = argument_parser.parse_args()

    main(args)
