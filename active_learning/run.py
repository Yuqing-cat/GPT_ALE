import os
import pandas as pd
import torch
from torch.utils.data import RandomSampler, DataLoader
import numpy as np
import json
import mlflow

# local imports
from helpers import request_annotations, check_for_annotations, lock, unlock, load_config, flatten_config, merge_args_w_config, dump_config
from data_utils import PandasDataset
from data_prep import align_w_sme, adjust_class_imbalance
from sampler import ATLSampler, ClusterBasedSampler, ErrorBasedSampler
from teacher import Teacher
from labeling import annotate_w_gpt3
from featurizer import Featurizer
from tsne import get_tsne, plot_tsne
from meters import Meter
import logging
import time
import argparse
import rpdb

def draw_sample(dataset, where=(), n_samples=1000, unequal=False):
    """
    Returns a list of indices of samples.

    Parameters:
    -----------
    dataset: PandasDataset
        Dataset from which samples are drawn.
    where: tuple
        Tuple of length 2. First element is the column name, second element is the value.
    n_samples: int
        Number of samples to draw.

    """
    if len(where) == 0:
        return np.random.choice(dataset.df.index, n_samples, replace=False)
    elif len(where) == 2:
        if unequal == False:
            candidates = np.where(dataset.df[where[0]].values == where[1])[0]
        else:
            candidates = np.where(dataset.df[where[0]].values != where[1])[0]
    else:
        raise ValueError("where must be a tuple of length 2 (column, value)")

    n_samples = min(n_samples, len(candidates))
    candidates = np.random.choice(candidates, n_samples, replace=False)

    idx = np.array(dataset.df.index[candidates])
    
    return idx

def update_datasets(datasets):
    ltt, ttl, modified = datasets['all'].update_label_target_mapping()

    for key in ['all', 'test_all']:
        datasets[key].set_label_target_mapping(
            label_to_target=ltt,
            target_to_label=ttl)
        
    return modified

def progress_update(progress, config):
    with open(os.path.join(config["misc"]["api_path"], 'progress.log'), 'w') as f:
        f.write("%0.1f" % progress)
        
def main(config):
    iterations_i = config['misc']['iterations_i']
    iterations_o = config['misc']['iterations_o']
    batch_size = config['model']['batch_size']
    num_epochs = config['misc']['num_epochs']
    total_samples_sme = config['misc']['total_samples_sme']
    samples_per_iteration = config['misc']['samples_per_iteration']
    samples_per_test = config['misc']['samples_per_test']
    patience_inner_loop = config['misc']['patience_inner_loop']
    patience_outer_loop = config['misc']['patience_outer_loop']
    # setting the lock means job cannot be annotated
    lock(config)

    dataset = config['data']['dataset']
    subset = config['data']["subset"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # file path for storing the student model
    if subset is None:
        config['model']['model_weights_latest']  = os.path.join(config["misc"]["output_path"], dataset + "_student_latest.pt")
        config['model']['model_weights_best']  = os.path.join(config["misc"]["output_path"], dataset + "_student_best.pt")
    else:
        config['model']['model_weights_latest']  = os.path.join(config["misc"]["output_path"], dataset + "_" + subset + "_student_latest.pt")
        config['model']['model_weights_best']  = os.path.join(config["misc"]["output_path"], dataset + "_" + subset + "_student_best.pt")
    
    # Set the experiment
    mlflow.set_experiment("Palantir")
    # Start the run
    mlflow_run = mlflow.start_run()
    mlflow_info = mlflow_run.to_dictionary()['info']
    with open(os.path.join(config["misc"]["api_path"], 'mlflow_info.json'), 'w') as f:
        json.dump(mlflow_info, f)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    teacher = Teacher(config)
    featurizer = Featurizer(config)
    meter_val = Meter(config, name='val')
    meter_test = Meter(config, name='test')

    # define dataset and dataloader for training and validation
    datasets = {}
    dataloaders = {}
    samplers = {}
    # mapping of labels (class_names) to indices (numeric labels). These need to be the same for train and test set
    label_to_target = config['data']['label_dict']
    target_to_label = {v: k for k, v in label_to_target.items()}
    for split in ["train", "test"]:
        if subset is None:
            # filename = os.path.join(os.path.join(config['data']['data_path'], dataset + "old", "data_" + split + ".pkl"))
            filename = os.path.join(os.path.join(config['data']['data_path'], dataset, "data_" + split + "_proc.csv"))
        else:
            # filename = os.path.join(os.path.join(config['data']['data_path'], dataset + "_" + subset + "_old", "data_" + split + ".pkl"))
            filename = os.path.join(os.path.join(config['data']['data_path'], dataset + "_" + subset, "data_" + split + "_proc.csv"))

        df = pd.read_csv(filename)
        # enforce data types and defaults
        df = df.astype({'probs': 'object', 'score': 'float64', 'label':'object'})
        df['probs'] = None

        # df = pd.read_pickle(filename)
        adjust_class_imbalance(df, split, config)
        align_w_sme(df, config)

        if split == 'train':
            df['ann_by'] = ""
            df['sampler'] = ""
            ds_name = "all"
        else:
            ds_name = "test_all"

        datasets[ds_name] = PandasDataset(df, config, label_to_target, target_to_label)
        label_to_target, target_to_label = datasets[ds_name].label_to_target, datasets[ds_name].target_to_label
        datasets[ds_name].update_emb(featurizer)
        samplers[ds_name] = RandomSampler(datasets[ds_name], replacement=False, num_samples=samples_per_test)
        dataloaders[ds_name] = DataLoader(datasets[ds_name], batch_size=batch_size, sampler=samplers[ds_name])

    # define the aux model
    # the aux model uses the same embeddings as the main model as input, and tries to predict whether the main model got it right
    input_dim = datasets['all'].embedding_size
    hidden_dim = 2 # correct/incorrect
    atl_sampler = ATLSampler(input_dim, hidden_dim, config) #, model_aux, optimizer_aux, criterion)
    cluster_based_sampler = ClusterBasedSampler(config)
    error_based_sampler = ErrorBasedSampler(config)
    student_model = None

    acc_best_outer = 0
    acc_best_val = 0
    best_iter_outer = 0
    n_annotations = {'train': {'total' : 0}, 'val': {'total' : 0}}
    gpt_acc_s = None

    step = 0 # merely used for tensorboard logging
    # outer loop: SME annotates samples for validation set
    # inner loop: large language model annotates samples for training set
    # after we haven't seen any improvement in inner loop for a while, we pop out to outer loop and ask sme for more annotations for validation set
    for i_outer in range(iterations_o):
        
        # we load the best model for requesting annotations, to calc probs for tnse       
        if student_model is not None and os.path.exists(config['model']['model_weights_best']):
            student_model.load_state_dict(torch.load(config['model']['model_weights_best']))
            student_model = student_model.to(device)
            teacher.fitted = True

        #
        # ask sme for annotations ----------------------------
        #
        
        # reset all AL scores
        datasets['all'].update_df({"score": 0})

        # score based on atl (we draw entirely random samples, unannotated and annotated)
        if atl_sampler.fitted and config['atl_sampler']['enabled']:
            candidates = draw_sample(datasets['all'], n_samples=1000)
            datasets['candidates'] = torch.utils.data.Subset(datasets['all'], candidates)
            dataloaders['candidates'] = DataLoader(datasets['candidates'], shuffle=False, batch_size=batch_size, drop_last=False)
            scores = atl_sampler.score(dataloaders['candidates'])
            datasets['all'].update_df({"score": scores, "sampler": "atl"}, idx=candidates)

        # score based on distribution (only unannotated samples)
        if config['cluster_based_sampler']['enabled']:
            candidates = draw_sample(datasets['all'], ('ann', -1), n_samples=1000)
            if isinstance(candidates, np.ndarray) and candidates.size > 0:
                datasets['candidates'] = torch.utils.data.Subset(datasets['all'], candidates)
                dataloaders['candidates'] = DataLoader(datasets['candidates'], shuffle=False, batch_size=batch_size, drop_last=False)
                scores = cluster_based_sampler.score(datasets['candidates'], total_samples=total_samples_sme)
                datasets['all'].update_df({"score": scores, "sampler": "cluster"}, idx=candidates)
            else:
                logging.warn("No unannotated samples for cluster_based_sampler")

        # score annotated samples based on performance
        if teacher.fitted and config['error_based_sampler']['enabled']:
            candidates = draw_sample(datasets['all'], ('ann', -1), n_samples=1000, unequal=True)
            datasets['candidates'] = torch.utils.data.Subset(datasets['all'], candidates)
            dataloaders['candidates'] = DataLoader(datasets['candidates'], shuffle=False, batch_size=batch_size, drop_last=False)
            
            _, _, corrs, _, _, probs, confs, _, _ = teacher.eval(student_model, dataloaders['candidates'], criterion, return_loss=False)
            datasets['all'].update_df({'corr': corrs, "probs": probs.tolist(), "confs": confs.tolist()}, idx=candidates)

            scores = error_based_sampler.score(datasets['candidates'])
            datasets['all'].update_df({"score": scores, "sampler": "error"}, idx=candidates)

        # get all rows with score > 0 for tsne
        candidates = draw_sample(datasets['all'], ('score', 0), n_samples=1000, unequal=True)
        datasets['candidates'] = torch.utils.data.Subset(datasets['all'], candidates)
        dataloaders['candidates'] = DataLoader(datasets['candidates'], shuffle=False, batch_size=batch_size, drop_last=False)
        # add tsne
        confs, preds = None, None
        if teacher.fitted:
            _, _, _, _, _, probs, confs, preds, _= teacher.eval(student_model, dataloaders['candidates'], criterion, return_loss=False)
            datasets['all'].update_df({"probs": None, 'confs': None})
            datasets['all'].update_df({"probs": probs.tolist(), "confs": confs.tolist()}, idx=candidates)
        tsne_emb = get_tsne(datasets['candidates'], verbose=False)
        datasets['all'].update_df({"tsne": None})
        datasets['all'].update_df({"tsne": tsne_emb}, idx=candidates)


        # TODO probably want to add reference df here
        annotate_w_gpt3(datasets['candidates'], None, config, featurizer, n_samples=total_samples_sme, for_sme=True)
        request_annotations(datasets['candidates'], config, preds=preds, confs=confs)
        datasets['all'].log_anns(step, split='val', per_class=True, destination='api')

        new_annotations = False
        progress = 100
        progress_update(progress, config)
        unlock(config)
        new_annotations = check_for_annotations(datasets['all'], config, wait=True)
        lock(config)

        if new_annotations:
            progress = 0
            progress_update(progress, config)
            modified = update_datasets(datasets)
            student_model, optimizer_main = teacher.init_student(datasets['all'], config)

        # init variables for inner loop
        best_iter_inner = 0
        tried_training_from_scratch = False
        progress = 5
        progress_update(progress, config)

        for i_inner in range(iterations_i): # training hyper loop in the slides
            #
            # init inner loop -------------------------------
            #

            #
            # prepare creating soft labels for training student ---------------------
            #
            # select "very large" subset of candidate samples for annotation. these will be processed and the top candidates will be sent to annotation
            candidates = draw_sample(datasets['all'], where=('ann', -1), n_samples=1000)
            if candidates.size == 0:
                logging.warn("No candidates for training student. breaking out of inner loop.")
                break

            datasets['candidates'] = torch.utils.data.Subset(datasets['all'], candidates)
            dataloaders['candidates'] = DataLoader(datasets['candidates'], shuffle=False, batch_size=batch_size, drop_last=False)

            # re-init annotated val dataset and loader
            annotated_val_samples = np.where(datasets['all'].df['ann_by'].values == 'sme')[0]
            val_batch_size = min(len(annotated_val_samples), 1024)
            datasets['val'] = torch.utils.data.Subset(datasets['all'], annotated_val_samples)
            dataloaders['val'] = DataLoader(datasets['val'], shuffle=True, batch_size=val_batch_size, drop_last=False)
            n_annotations['val'] = datasets['all'].get_ann_counts(ann_by='sme')

            if n_annotations['train']['total'] == 0:
                scores = cluster_based_sampler.score(datasets['candidates'], total_samples=samples_per_iteration)
            else:    
                # re-init annotated train dataset and loader
                annotated_training_samples = np.where(datasets['all'].df['ann_by'].values == 'gpt3')[0]
                datasets['train'] = torch.utils.data.Subset(datasets['all'], annotated_training_samples)
                # train_batch_size = min(len(annotated_training_samples), 1024)
                samplers['train'] = RandomSampler(datasets['train'], replacement=True, num_samples=batch_size * 10)
                dataloaders['train'] = DataLoader(datasets['train'], sampler=samplers['train'], batch_size=batch_size)
            
                _, corr, probs = teacher.run(student_model, dataloaders, optimizer_main, criterion, num_epochs=num_epochs)
                meter_val.set_epoch(i_outer)
                # add corr to dataset so we can train ATL
                datasets['all'].update_df({'corr': corr, "probs": probs.tolist()}, idx=annotated_val_samples)

                # progress_update(30, config)

                # train teacher model
                datasets['all'].selected_columns = ['embedding', 'corr']
                if i_inner % 2 == 0:
                    re_init = True
                else:
                    re_init = False
                atl_sampler.update(datasets['val'], re_init=re_init)
                
                datasets['all'].selected_columns = ['embedding', 'ann']

                # apply aux model to score each candidate for how valueable annotating them would be
                scores = atl_sampler.score(dataloaders['candidates'])

            #
            # create soft labels for training student
            #
            datasets['all'].update_df({"score": 0})
            try:
                datasets['all'].update_df({"score": scores}, idx=candidates)
            except IndexError:
                logging.warn("No candidates for annotation.")
                rpdb.set_trace()

            # 
            # evaluate on test set ------------------------
            # 
            logging.debug("Testing student model.")
            datasets['test_all'].selected_columns = ['embedding', 'target']
            _, acc_main_test, _, _, _, _, _, _, sentences_per_second = teacher.eval(student_model, dataloaders['test_all'], criterion, meter=meter_test, return_loss=False)

            meter_test.log(step)
                        
        
            if n_annotations['train']['total'] == 0:
                # if we have saved best model, load it
                # we are using the latest model here, because we may want to keep training that one with more data
                if os.path.exists(config['model']['model_weights_latest']):    
                    student_model.load_state_dict(torch.load(config['model']['model_weights_latest']))
                # apply main model to annotation candidates (we need the probs for annotation below)
                _, _, _, _, _, probs, confs, preds, sentences_per_second = teacher.eval(student_model, dataloaders['candidates'], criterion, return_loss=False)
                datasets['all'].update_df({"probs": probs.tolist()}, idx=candidates)

            # GPT3 annotates the selected candidates. (this uses score, to select samples to annotate)
            annotate_w_gpt3(datasets['all'], datasets['val'], config, featurizer, n_samples=samples_per_iteration)
            
            # evalute quality of gpt-3  annotations
            gpt_acc_train = datasets['all'].eval_annotations(ann_by='gpt3')
            # evalute quality of sme  annotations
            gpt_acc_val = datasets['all'].eval_annotations(ann_by='sme')

            if gpt_acc_s is None:
                gpt_acc_s = gpt_acc_val
            else:
                gpt_acc_s = 0.99 * gpt_acc_s + 0.01 * gpt_acc_val


            # save intermediate df
            if SAVE_DF:
                suffix = "_o%02d_i%02d_train_a" % (i_outer, i_inner)
                datasets['all'].save(suffix=suffix, idx=candidates)

            # 
            # evaluate on test set ------------------------
            # 
            logging.debug("Validating student model.")
            # if we have saved latest model, load it. we are using the latest model here, because we want to compare it to the best model
            if os.path.exists(config['model']['model_weights_latest']):    
                student_model.load_state_dict(torch.load(config['model']['model_weights_latest']))
            _, acc_main_val, _, _, _, _, _, _, sentences_per_second = teacher.eval(student_model, dataloaders['val'], criterion, meter=meter_val, return_loss=False)
            meter_val.to_json()

            # --- mlflow logging --- 
            n_annotations['train'] = datasets['all'].get_ann_counts(ann_by='gpt3')
            datasets['all'].log_anns(step, split='train')

            n_annotations['val'] = datasets['all'].get_ann_counts(ann_by='sme')
            datasets['all'].log_anns(step, split='val')

            datasets['all'].log_samplers(step, split='val')

            mlflow.log_metric('student/val/acc_latest', acc_main_val, step)
            mlflow.log_metric('student/test/acc', acc_main_test, step)
            mlflow.log_metric("gpt3/val/acc", gpt_acc_val, step)
            mlflow.log_metric("gpt3/train/acc", gpt_acc_train, step)
            
            # compare current accuracy with best accuracy
            # we the latest model is better than the previous best model, we replace the best model with the latest model
            if acc_main_val > acc_best_val:
                acc_best_val = acc_main_val
                best_iter_inner = i_inner
                mlflow.log_metric("student/val/acc_best", acc_best_val, step)        
                torch.save(student_model.state_dict(), config['model']['model_weights_best'])

            # if it has been too long since we've seen an improvement, we stop
            if i_inner - best_iter_inner > patience_inner_loop or acc_best_val == 1:
                if tried_training_from_scratch == False and acc_best_val < 1:
                    logging.info("trying to train student from scratch, before popping out of inner loop")
                    student_model, optimizer_main = teacher.init_student(datasets['all'], config)
                    tried_training_from_scratch = True
                else:
                    logging.info("popping out of inner loop to ask SME for more data")
                    break
            
            try:
                progress = min(95, max(progress, (i_inner - best_iter_inner) / patience_inner_loop * 100))
            except ZeroDivisionError:
                progress = 95

            if tried_training_from_scratch == False:
                progress *= .5
            progress_update(progress, config)


            step += 1

        if acc_best_val > acc_best_outer:
            acc_best_outer = acc_best_val
            best_iter_outer = i_outer
            mlflow.log_metric("student/test/acc_best_o", acc_best_outer, i_outer)

        # if we have not improved for a while, despite more data from SME, we give up.
        # this should have a very high patience, because we want to make sure we are not upsetting sme
        if i_outer - best_iter_outer > patience_outer_loop:
            print("stopping early, after %d outer iterations (best iteration: %d)" % (i_outer, best_iter_outer))
            break

    # add some additional stats to the results
    meter_test.metrics['misc'] = {}
    meter_test.metrics['misc']['n_annotations_train'] = n_annotations['train']['total']
    meter_test.metrics['misc']['n_annotations_val'] = n_annotations['val']['total']
    meter_test.metrics['misc']['gpt_acc'] = gpt_acc_s
    meter_test.metrics['misc']['best_iter_outer'] = best_iter_outer
    meter_test.metrics['misc']['acc_best_outer'] = acc_best_outer
    meter_test.metrics['misc']['i_outer'] = i_outer
    meter_test.metrics['misc']['sentences_per_second'] = sentences_per_second


    return meter_test

if __name__ == "__main__":
    import sys
    print("sys.argv", sys.argv)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_path', type=str, default='configs/dbpedia_14/0')
    argparser.add_argument("--config", nargs="+", help="key-value pair for configuration settings")
    args = argparser.parse_args()
    print(args)
    # set this to true if you want intermediate results to be saved as csv files in the results directory
    SAVE_DF = False

    config = load_config(args.config_path)
    config = merge_args_w_config(config, args)
    dump_config(config)

    if os.path.exists(config['misc']['output_path']) == False:
        os.makedirs(config['misc']['output_path'])
    log_filename = os.path.join(config["misc"]["output_path"], 'run.log')
    # log_filename = 'run.log'
    if config['misc']['log_level'] == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(filename=log_filename, level=log_level, filemode='w')
    logging.info("starting run.py")

    start_time = time.time()    
    meter = main(config)
    run_time = time.time() - start_time

    config_cp = dict(config)
    for key in ['openai', 'sql']:
        config_cp.pop(key)

    # get global metrics
    meter_final = {k: max(v[1]) for k, v in meter.metrics['global'].items()}

    # get misc metrics
    meter_final['misc'] = {k: v for k, v in meter.metrics['misc'].items()}
    meter_final['misc']['run_time'] = run_time

    # save results
    config_flat = flatten_config(config_cp)
    results = {**config_flat, **meter_final}
    if os.path.exists(os.path.join(config["misc"]["output_path"], 'results.pkl')):
        df = pd.read_pickle(os.path.join(config["misc"]["output_path"], 'results.pkl'))
        df = pd.concat([df, pd.DataFrame(results, index=[0])])
    else:
        df = pd.DataFrame(results, index=[0])
    df.to_pickle(os.path.join(config["misc"]["output_path"], 'results.pkl'))

    # End run 
    mlflow.end_run()
