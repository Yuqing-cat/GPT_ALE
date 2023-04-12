import mlflow
import numpy as np
import sklearn.metrics
import os
import json

class Meter():
    def __init__(self, config, name='test', avg='macro', metrics=['accuracy_score', 'precision_score', 'recall_score', 'f1_score']):
        self.config = config
        self.name = name
        self.metrics = {}
        self.metrics['global'] = {metric: [] for metric in metrics}
        self.metrics['confusion_matrix'] = {}
        self.methods = {}
        for metric in metrics:
            self.methods[metric] = sklearn.metrics.__dict__[metric]
        self.avg = avg
        self.no_averaging = ['accuracy_score'] # some methods don't support averaging
        self.return_metric = metrics[0]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update(self, targets, preds, dataset):
        uq_targets = np.unique(targets)

        self.calculate_cms(targets, preds)

        for metric in self.methods:
            if metric in self.no_averaging:
                self.metrics['global'][metric].append((self.epoch, self.methods[metric](targets, preds)))
            else:
                self.metrics['global'][metric].append((self.epoch, self.methods[metric](targets, preds, average=self.avg, zero_division=0)))
        
        # calculate per class metrics
        for uq_target in uq_targets:

            class_name = dataset.target_to_label.get(uq_target, 'unknown')
            if class_name == "unknown":
                continue

            # if this is the first time we've seen this class, initialize it
            if class_name not in self.metrics:
                # if other classes have been seen, initialize this class with prefix of 0s
                if len(self.metrics['global'][list(self.methods.keys())[0]]) > 1:
                
                    prefix = [(e, 0) for e, _ in self.metrics['global'][list(self.methods.keys())[0]][:-1]]
                    self.metrics[class_name] = {metric: list(prefix) for metric in self.methods}
                else:
                    self.metrics[class_name] = {metric: [] for metric in self.methods}
            # get a mask of the targets that are this class
            mask_target = targets == uq_target
            mask_pred = preds == uq_target
            for metric in self.methods:
                # skip class if there are no targets of this class
                if mask_target.sum() > 0:
                    # if the metric doesn't support averaging, don't average
                    if metric in self.no_averaging:
                        self.metrics[class_name][metric].append((self.epoch, self.methods[metric](mask_target, mask_pred)))
                    else:
                        self.metrics[class_name][metric].append((self.epoch, self.methods[metric](mask_target, mask_pred, zero_division=0)))
                else:
                    self.metrics[class_name][metric].append((self.epoch, self.metrics[uq_target][metric][-1]))
        
        # for each target in the dataset that is not in the current test set, we copy the previous value
        for target in dataset.target_to_label.keys():
            if target not in self.metrics.keys():
                continue
            if target not in uq_targets:
                class_name = dataset.target_to_label[target]
                for metric in self.methods:
                    self.metrics[class_name][metric].append((self.epoch, self.metrics[class_name][metric][-1]))

        return self.metrics['global'][self.return_metric][-1][1]

    def calculate_cms(self, targets, preds):
        self.metrics['confusion_matrix']['none'] = sklearn.metrics.confusion_matrix(targets, preds).tolist()

        for norm in ["true", "pred", "all"]:
            self.metrics['confusion_matrix'][norm] = sklearn.metrics.confusion_matrix(targets, preds, normalize=norm).tolist()

    def to_json(self):
        with open(os.path.join(self.config['misc']['output_path'], 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)

    def log(self, step, metrics=['f1_score']):
        for category in self.metrics:
            if category in ['global', 'confusion_matrix']:
                continue
            for metric in metrics:
                mlflow.log_metric(f'student/{self.name}/{metric}/{category}', self.metrics[category][metric][-1][1], step)
