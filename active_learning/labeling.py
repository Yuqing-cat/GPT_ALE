

import json
import logging
import time
import numpy as np

import requests
import torch


def generate_prompt_v1(prompt_header="Please classify the following text sentences", second_prompt="", labeled_samples=[], new_sample=""):
    if prompt_header != "":
        prompt = prompt_header + "\n"
    else:
        prompt = ""

    for sample, label in labeled_samples:
        prompt += f"\n{sample}\nCategory: {label}\n"
    if len(labeled_samples) > 0:
        prompt += "\n"

    if second_prompt != "":
        prompt += f"{second_prompt}\n"

    prompt += f"{new_sample}\nCategory:"

    return prompt

def generate_prompt_v2(prompt_header="Please classify the following text sentences", second_prompt=None, labeled_samples=[], new_sample=""):
    prompt = ""
    for sample, label in labeled_samples:
        prompt += f"{prompt_header}\nSentence: {sample}\nCategory: {label}\n\n"

    prompt += f"{prompt_header}\nSentence: {new_sample}\nCategory:"

    return prompt

# curl https://azopenaipocs.openai.azure.com/openai/deployments/text-curie-001/completions?api-version=2022-06-01-preview\
#   -H "Content-Type: application/json" \
#   -H "api-key: 556266146da8462185bd4002c545fb23" \
#   -d "{
#   \"prompt\": \"Once upon a time\",
#   \"max_tokens\": 5
# }"

def make_call(prompts, config):
    max_attempts = 10
    response = None
    
    for attempt in range(max_attempts):
        try:
            data = json.dumps(
                {"prompt": prompts,
                "temperature": 0,
                "max_tokens": 10,
                "top_p":1.0,
                "frequency_penalty":0.0,
                "presence_penalty":0.0,
                "stop":["\n",",","."]})
            response = requests.post(
                config['openai']['endpoint'],
                data=data,
                headers={"Content-Type": "application/json", "api-key": config["openai"]['api_key']})
            return response.json()['choices']

        except Exception as e:
            print(e)
            delay = attempt
            print("retrying in %d seconds."  % delay)
            time.sleep(delay)

    return False
    
def find_closest_label(dataset, label, featurizer, target_to_label):
    emb = featurizer.featurize(label)
    try:
        df = dataset.df # pandas dataframe with unlabeled data
    except AttributeError as e:
        df = dataset.dataset.df

    sim = torch.cosine_similarity(emb.cpu(), df.embeddings, dim=1) # calculate cosine similarity between sample and labeled data of top choice
    
    return target_to_label[sim.argmax().item()]
    

def permute_anns(gt, config):
    percentage = config['misc']['gpt3_error_rate']

    mask = [x['text'] != "unknown" for x in gt]
    subset = [x for x, m in zip(gt, mask) if m]
    permuted_indices = np.random.permutation(len(subset))
    permuted_indices = permuted_indices[:int(len(subset) * percentage)]
    for i, j in zip(permuted_indices, np.random.permutation(subset)):
        gt[mask.index(True, i)] = j
        
def annotate_w_gpt3(dataset_for_annotation, dataset_for_reference, config, featurizer, n_samples=50, for_sme=False, delay=1):
    logging.info('Talking to GPT')
    
    if for_sme:
        gpt3_enabled = config["misc"]['use_gpt3_for_sme']
    else:
        gpt3_enabled = config["misc"]['use_gpt3_for_student']

    try:
        api_key = config["openai"]['api_key']
    except KeyError:
        if gpt3_enabled:
            logging.warn("GPT3 is enabled but no API key is provided. Disabling GPT3.")
        api_key = ""
        gpt3_enabled = False

    try:
        df_ann = dataset_for_annotation.df # pandas dataframe with unlabeled data
        target_to_label = dataset_for_annotation.target_to_label
        label_to_target = dataset_for_annotation.label_to_target
    except AttributeError as e:
        df_ann = dataset_for_annotation.dataset.df
        target_to_label = dataset_for_annotation.dataset.target_to_label
        label_to_target = dataset_for_annotation.dataset.label_to_target

    labels = list(target_to_label.values())
    labels.remove("unknown")

    if dataset_for_reference is None:
        df_ref = None
    else:
        try:
            df_ref = dataset_for_reference.df # df with already labeled data
        except AttributeError as e:
            df_ref = dataset_for_reference.dataset.df

    if for_sme:
        df_ann['gpt3'] = "unknown"

    # progress = 90.0
    # with open(os.path.join(config["data"]["data_path"], 'progress.log'), 'w') as f:
    #     f.write("%0.1f" % progress)

    topk = df_ann.nlargest(n_samples, "score") # get most "beneficial" samples for annotation (aux)
    prompts = []
    indices = []
    results = []
    gt_labels = []
    for i, row in topk.iterrows():
        labeled_samples = []
        prompt_header = "Please Classify the following sentences."
        second_prompt = prompt_header
        if df_ref is not None:
            if row.probs is not None:
                probs = json.loads(row['probs']) # class probabilities the main model assigned to the sample
            else:
                # we create random probabilities if the main model did not assign any (before training model for the first time)
                probs = np.random.uniform(0, 1 / dataset_for_annotation.n_classes, size=dataset_for_annotation.n_classes)
            if isinstance(row['embedding'], list):
                emb = torch.tensor(row['embedding'], dtype=torch.float)
            else:
                emb = torch.tensor(json.loads(row['embedding']), dtype=torch.float)
            probs_ranked = np.argsort(probs)[::-1]

            proposed_labels = [target_to_label.get(idx, None) for idx in probs_ranked]
            proposed_labels = set(proposed_labels)
            if None in proposed_labels:
                proposed_labels.remove(None)
            proposed_labels = list(proposed_labels)

            # remove duplicates
            # proposed_labels = list(dict.fromkeys(proposed_labels))
            # if "Other" in proposed_labels:
            #     proposed_labels.remove("Other")
            # Example: Categorize the next sentence into one of these: "Politician", "Artist". If those categories don't fit well, use "Other".
            # prompt_header = "Categorize the next sentence into one of these: \"%s\". If those categories don't fit well, use \"Other\"" % ("\", \"".join(proposed_labels))
            second_prompt = "Classify the following sentence into one of these categories: \"%s\"." % ("\", \"".join(proposed_labels))

            annotated_samples = np.where(df_ref['ann'] != -1)[0]
            sample_size = min(annotated_samples.shape[0], 500)
            idx = np.random.choice(annotated_samples, size=sample_size, replace=False)
            ref_emb = torch.zeros((sample_size, emb.shape[0]), dtype=torch.float)
            for j, idx_j in enumerate(idx):
                ref_emb[j] = torch.tensor(json.loads(df_ref.loc[idx_j, 'embedding']), dtype=torch.float)

            sim = torch.cosine_similarity(emb, ref_emb, dim=1) # calculate cosine similarity between sample and labeled data of top choice

            closest_embeddings = torch.topk(sim, 4).indices.numpy().tolist() # get the indices of the two most similar samples

            ref = df_ref.iloc[idx[closest_embeddings]] # get the actual samples from the df

            for j, row_j in ref.iterrows():
                labeled_samples.append([row_j['text'].strip(), str(target_to_label.get(row_j["ann"], "unknown"))])
 
        prompt = generate_prompt_v1(prompt_header=prompt_header, second_prompt=second_prompt, labeled_samples=labeled_samples, new_sample=row['text'].strip())
        prompts.append(prompt)
        indices.append(i)
        if "target" in row:
            gt_labels.append({'text': target_to_label.get(row["target"], "unknown")})
        else:
            gt_labels.append({'text': "unknown"})

        if len(prompts) % 20 == 0:
            if api_key == "" or gpt3_enabled == False:
                results.append((list(indices), list(prompts), list(gt_labels)))
            else:
                result = make_call(prompts, config)
                results.append((list(indices), list(prompts), result))
                time.sleep(delay)
            indices = []
            prompts = []
            gt_labels = []

    if len(prompts) > 0:
        if api_key == "" or gpt3_enabled == False:
            results.append((list(indices), list(prompts), list(gt_labels)))
        else:
            result = make_call(prompts, config)
            results.append((list(indices), list(prompts), result))
            time.sleep(delay)

    for result in results:
        for i, prompt, choice in zip(result[0], result[1], result[2]):
            if choice is None or choice['text'].strip() == "":
                logging.warning("no result. \n Prompt: %s\n Choice: %s" % (prompt, choice))
            else:
                label = choice['text'].strip()
                closest_label = ""
                # if we are creating soft labels, we write to the 'ann' column
                # 'unknown' is a special label that is used to indicate that the sample should not be used for training
                if label in label_to_target.keys() and not for_sme and label != "unknown":
                    if config['misc']['gpt3_error_rate'] > 0:
                        # simualte gpt3 errors
                        if np.random.uniform() < config['misc']['gpt3_error_rate']:
                            other_labels = [l for l in labels if l != label]
                            if len(other_labels) > 0:
                                label = np.random.choice(other_labels)
                    df_ann.loc[i, 'ann'] = label_to_target[label]
                    df_ann.loc[i, 'ann_by'] = 'gpt3'
                    if 'target' in df_ann.columns and df_ann.loc[i, 'target'] != df_ann.loc[i, 'ann']:
                        with open("gpt_debug.log", "a") as f:
                            # f.write("sample: " + df_ann.loc[i, 'content'] + "\n")
                            f.write("prompt: " + prompt + "\n")
                            f.write("ann: " + label + "\n")
                            f.write("closest: " + closest_label + "\n")
                            f.write("target: " + df_ann.loc[i, 'label'] + "\n\n")
                        
                # if we are creating suggestions for SME, we write to 'gpt3'
                elif for_sme and len(label) > 0:
                    logging.debug("keeping ann: %s\n, prompt: %s" % (label, prompt))
                    df_ann.loc[i, 'gpt3'] = label
                    # df_ann.loc[i, 'ann'] = len(label_to_target) + len(new_labels)
                    # if label not in new_labels:
                    #     new_labels.append(label)
                    # try:
                    #     df_ann.loc[i, 'ann'] = label_to_target[label]
                    # except KeyError as e:
                    #     # note, we set this to 0, with the expectation that the SME will correct it
                    #     df_ann.loc[i, 'ann'] = -1
                # elif use_closest_label:
                #     closest_label = find_closest_label(dataset_for_annotation, label, featurizer, target_to_label)
                #     df_ann.loc[i, 'ann'] = label_to_target[closest_label]
                    # you can examine the issue by printing the prompt and the label
                    # very often, the two top choices category choices where just not even close to the correct answer
                    # this will happen a lot when the main model has low peformance at the beginning of training
                else:
                    logging.debug("dropping ann")
                # if label != df_ann.loc[i, 'label']:
                    with open("gpt_debug.log", "a") as f:
                        # f.write("sample: " + df_ann.loc[i, 'content'] + "\n")
                        f.write("prompt: " + prompt + "\n")
                        f.write("ann: " + label + "\n")
                        f.write("closest: " + closest_label + "\n")
                        f.write("target: " + df_ann.loc[i, 'label'] + "\n\n")
    
    # progress = 100.0
    # with open(os.path.join(config["data"]["data_path"], 'progress.log'), 'w') as f:
    #     f.write("%0.1f" % progress)
