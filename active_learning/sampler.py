import os
import torch
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random
import torch
import json
import time
import torch.nn.functional as F
import logging
import numpy as np
from teacher import MLP
import subprocess 
import mlflow

class Sampler(object):
    def update():
        raise NotImplementedError
    
    def score():
        raise NotImplementedError

class ATLSampler(Sampler):
    def __init__(self, input_dim, hidden_dim, config):
        super(ATLSampler, self).__init__()
        self.config = config
        self.alpha = self.config['atl_sampler']['alpha']
        self.max_epoch = self.config['atl_sampler']['max_epochs']
        self.patience = self.config['atl_sampler']['patience']
        self.tolerance = self.config['atl_sampler']['tolerance']
        self.batch_size = self.config['atl_sampler']['batch_size']
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.config['atl_sampler']['output_dim']
        self.fitted = False

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.init_model()

    def init_model(self):
        # print("init model", self.input_dim, self.hidden_dim, self.output_dim)
        self.model = MLP(self.input_dim, self.hidden_dim, self.output_dim)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['atl_sampler']['learning_rate'], weight_decay=self.config['atl_sampler']['weight_decay'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")


    def update(self, dataset, re_init=False):
        if re_init:
            self.init_model()

        # val_size = max(1, len(dataset) // 10)
        # train_size = len(dataset) - val_size

        df_s = dataset.dataset.df[dataset.dataset.df['ann_by'] == 'sme']
        corrs = df_s['corr'].values

        # do stratified sampling, if we have at least to samples of each class
        values, counts = np.unique(corrs, return_counts=True)
        stratify = corrs if min(counts) > 1 else None

        train_idx, valid_idx = train_test_split(
            np.arange(len(corrs)),
            test_size=0.1,
            shuffle=True,
            stratify=stratify)

        sampler_train = torch.utils.data.SubsetRandomSampler(train_idx)
        sampler_val = torch.utils.data.SubsetRandomSampler(valid_idx)

        dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler_train)
        dataloader_val = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler_val)

        # classweights: for correct samples, weight is 1 - mean_acc, for incorrect samples, weight is mean_acc
        # i.e. basically the inverse of the performance
        class_weights = torch.Tensor([df_s['corr'].mean(), 1 - df_s['corr'].mean()]).to(self.device)

        best_loss = torch.inf
        best_epoch = 0
        logging.info("Teacher model training")
        start_time = time.time()
        for epoch in range(self.max_epoch):
            self.model.train()
            for i, sample in enumerate(dataloader_train):
                data = sample[0].to(self.device)
                targets = sample[1].to(self.device).to(torch.int64)
                outputs = self.model(data)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets).mean()
                loss_weights = class_weights[targets]
                if torch.max(loss_weights).item() == 1:
                    logging.warning("There is a Loss weight of 1. This is probably not intended.")
                loss = (loss_weights * loss).mean()
                loss.backward()
                self.optimizer.step()
                
                mlflow.log_metric("teacher/train/loss", loss)

            self.model.eval()
            with torch.no_grad():
                f1scores = []
                val_losses = []
                for i, sample in enumerate(dataloader_val):
                    data = sample[0].to(self.device)
                    targets = sample[1].to(self.device).to(torch.int64)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets).mean()
                    val_losses.append(loss.item())

                    preds = torch.argmax(outputs.detach(), dim=1)
                    f1scores.append(f1_score(targets.cpu(), preds.cpu(), average='macro'))

            val_loss = np.mean(val_losses)
            f1score = np.mean(f1scores)

            mlflow.log_metric("teacher/val/loss", val_loss)
            mlflow.log_metric("teacher/val/f1", f1score)
            
            if val_loss < best_loss - self.tolerance:
                best_loss = val_loss
                best_epoch = epoch

            if epoch - best_epoch > self.patience:
                break
        
        self.fitted = True

        logging.debug("total time: %0.2fs (%d epochs)" % (time.time() - start_time, epoch))

    def score(self, dataloader):
        logging.debug("ATL sampling")
        logging.info("Organizing data")
        scores = torch.zeros(len(dataloader.dataset), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                data = sample[0].to(self.device)
                outputs = self.model(data)
                probs = softmax(outputs, dim=1)
                curr_batch_size = probs.shape[0]
                scores[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = probs[:,0]

        scores /= torch.sum(scores)

        return scores.cpu().numpy()


class ClusterBasedSampler(Sampler):
    def __init__(self, config):
        super(ClusterBasedSampler, self).__init__()
        self.config = config
        self.n_centroid = self.config['cluster_based_sampler']['n_centroid']
        self.n_outlier = self.config['cluster_based_sampler']['n_outlier']
        self.n_random = self.config['cluster_based_sampler']['n_random']
        self.max_epochs = self.config['cluster_based_sampler']['max_epochs']
    
    def score(self, dataset, total_samples=75):
        """ This (incl CostineClusters and Cluster classes below) is based on 
        https://github.com/rmunro/pytorch_active_learning/blob/65e7c9c56d5e1f8a124fec5294f1d21733575339/diversity_sampling.py"""

        logging.debug("Cluster-based sampling")
        if hasattr(dataset, 'indices'):
            df = dataset.dataset.df.iloc[dataset.indices]
        else:
            df = dataset.df

        # data = [[textid, json.loads(embedding)] for textid, embedding in zip(df.index, df.embedding)]
        data = [[i, json.loads(embedding)] for i, embedding in enumerate(df.embedding)]
        num_clusters = total_samples // (self.n_centroid + self.n_outlier + self.n_random)

        kmeans = KMeans(data, self.config, num_clusters, max_epochs=self.max_epochs)
        kmeans.fit()

        centroids, outliers, randoms = kmeans.get_samples(n_random=self.n_random)
        
        indices = centroids + outliers + randoms

        scores = np.random.uniform(0, .5, df.shape[0])
        scores[indices] = np.random.uniform(.5, 1, len(indices))

        scores /= scores.sum()

        return scores


class KMeans():
    """
    Represents a set of clusters over a dataset
    
    """
    
    def __init__(self, data, config, num_clusters=15, max_epochs=2):
        self.num_clusters = num_clusters
        self.max_epochs = max_epochs
        self.config = config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_tensor = torch.zeros(len(data), len(data[0][1]))
        self.cluster_tensor = torch.zeros(num_clusters, len(data[0][1])).to(device)
        self.cluster_membership = torch.randint(0, num_clusters, (len(data),)).to(device)
        self.indices = torch.tensor([item[0] for item in data]).to(device)
    
        for i, item in enumerate(data):
            self.feature_tensor[i] = torch.tensor(item[1])
        self.feature_tensor = self.feature_tensor.to(device)

        for c in range(num_clusters):
            self.update_cluster_tensor(c)

    def update_cluster_tensor(self, c):
        if (self.cluster_membership == c).sum().item() > 0:
            self.cluster_tensor[c] = torch.median(self.feature_tensor[self.cluster_membership == c], dim=0)[0]

    def fit(self):
        logging.info("Data Clustering")
        start = time.time()
        for epoch in range(self.max_epochs):
            added = self.step()
            logging.debug("epoch_time: %f, added: %d" % ((time.time() - start)/ (epoch+1), added))
            
            if added == 0:
                break


        logging.debug("total time: %f (%d epochs)" % (time.time() - start, epoch))

    def step(self):
        added = 0

        for i in range(len(self.cluster_membership)):
            current_cluster = self.cluster_membership[i].item()
            self.cluster_membership[i] = -1
            self.update_cluster_tensor(current_cluster)
            item_tensor = self.feature_tensor[i]

            similarities = F.cosine_similarity(item_tensor, self.cluster_tensor, 1)

            closest = torch.argmax(similarities).item()

            self.cluster_membership[i] = closest
            self.update_cluster_tensor(closest)
            if current_cluster != closest:
                added += 1

        return added

    def get_samples(self, n_random=3):  
        centroids = []
        outliers = []
        randoms = []
        remaining_budget = 0
        for c in range(self.num_clusters):
            centroid, outlier = self.centroid_outlier(c)
            
            centroids.append(centroid)
            outliers.append(outlier)
            indices = self.indices[self.cluster_membership == c].cpu().numpy().tolist()
            if centroid in indices:
                indices.remove(centroid)
            if outlier in indices:
                indices.remove(outlier)
            if len(indices) < n_random:
                n_random_c = len(indices)
                remaining_budget += len(indices)
            else:
                n_random_c = n_random
            randoms += np.random.choice(indices, n_random_c, replace=False).tolist()

        # most likely one of the clusters is too small, so we add some random samples
        if remaining_budget > 0:
            remaining_indices = self.indices.cpu().numpy().tolist()
            for i in centroids + outliers + randoms:
                if i in remaining_indices:
                    remaining_indices.remove(i)
            randoms += np.random.choice(remaining_indices, remaining_budget, replace=False).tolist()

        return centroids, outliers, randoms
    
    def centroid_outlier(self, c):
        cluster_tensor = self.cluster_tensor[c]
        cluster_indices = torch.where(self.cluster_membership == c)[0]
        item_vectors = self.feature_tensor[cluster_indices]

        similarities = F.cosine_similarity(cluster_tensor, item_vectors, 1)

        centroid = torch.argmax(similarities).item()
        outlier = torch.argmin(similarities).item()

        return cluster_indices[centroid].item(), cluster_indices[outlier].item()


class StratifiedSampler(Sampler):
    def __init__(self):
        super(StratifiedSampler, self).__init__()

    def score(self, dataset, total_samples):
        logging.debug("Performing diversity sampling")
        
        if hasattr(dataset, 'indices'):
            df = dataset.dataset.df.iloc[dataset.indices]
        else:
            df = dataset.df

        scores = np.random.uniform(0, .5, df.shape[0])

        categories = df['label'].unique()
        n_categories = len(categories)

        samples_per_class = total_samples // n_categories

        for category in categories:
            indices = np.where(df['label'] == category)[0]
            indices = np.random.choice(indices, samples_per_class, replace=False)
            scores[indices] += .5

        scores /= scores.sum()

        return scores


class ErrorBasedSampler(Sampler):
    """
    Samples based on the confidence of the model. Samples that the model is most confident about, but wrong are sampled more.
    """
    def __init__(self, config):
        super(ErrorBasedSampler, self).__init__()
        self.config = config

    def score(self, dataset):
        logging.debug("Performing error based sampling")

        if hasattr(dataset, 'indices'):
            df = dataset.dataset.df.iloc[dataset.indices]
        else:
            df = dataset.df

        scores = np.zeros(df.shape[0])

        idx = np.where(df['corr'].values == 0)[0]

        scores[idx] = df['confs'].values[idx].copy()

        scores /= scores.sum()

        return scores
