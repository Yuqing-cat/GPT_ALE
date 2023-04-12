import os
import time
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
import logging
import numpy as np
import mlflow
import torch
from meters import Meter
from temperature_scaling import TemperatureScaling

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=1.0):
        super(MLP, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        outputs = self.layers(x)
        return outputs

    def set_temperature(self, temperature):
        self.temperature.data.fill_(temperature)


class Teacher(object):
    def __init__(self, config):
        self.fitted = False
        self.config = config
        self.ts = TemperatureScaling(config)

    def train(self, model, loader, optimizer, criterion):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        class_weights = torch.Tensor(loader.dataset.dataset.get_class_weights()).to(device)
        model.train()
        step = 0
        for features, labels in tqdm(loader, disable=True):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss_weights = class_weights[labels]
            if torch.max(loss_weights).item() == 1:
                logging.warning("There is a Loss weight of 1. This is probably not intended.")
            loss = (loss_weights * loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).sum().item() / len(labels)
            step += 1
        return loss, acc
        
    def eval(self, model, dataloader, criterion, meter=None, return_loss=True):
        losses = []

        try:
            dataset = dataloader.dataset.dataset
        except AttributeError as e:
            dataset = dataloader.dataset
        n_classes = model.layers[-1].out_features
        
        outputs_epoch = torch.zeros((len(dataloader) * dataloader.batch_size, n_classes), dtype=torch.float64)
        preds_epoch = torch.zeros((len(dataloader) * dataloader.batch_size), dtype=torch.int64)
        confs_epoch = torch.zeros((len(dataloader) * dataloader.batch_size), dtype=torch.float64)
        targets_epoch = torch.zeros((len(dataloader) * dataloader.batch_size), dtype=torch.int64)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.eval()
        duration = 0
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                data = sample[0].to(device)
                targets = sample[1].to(device)                
                # targets[torch.where(targets == -1)] = 0
                start_time = time.time()
                outputs = model(data)
                # probs = softmax(outputs, dim=1)
                duration += time.time() - start_time
                # confs, preds = torch.max(probs, dim=1) 
                if return_loss:
                    loss = criterion(outputs, targets).mean()
                    losses.append(loss.detach().cpu().item())
                curr_batch_size = len(targets) # last batch may be smaller
                outputs_epoch[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = outputs.detach()
                # probs_epoch[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = probs.detach().cpu().numpy()
                targets_epoch[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = targets.detach()
                # preds_epoch[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = preds.detach().cpu().numpy()
                # confs_epoch[i * dataloader.batch_size:i * dataloader.batch_size + curr_batch_size] = confs.detach().cpu().numpy()
        

        probs_epoch = softmax(outputs_epoch / self.ts.temperature.detach(), dim=1)
        confs_epoch, preds_epoch = torch.max(probs_epoch, dim=1)

        sentences_per_second = targets_epoch.shape[0] / duration
        crop_size = i * dataloader.batch_size + targets.shape[0]
        preds_epoch = preds_epoch[:crop_size]
        targets_epoch = targets_epoch[:crop_size]
        confs_epoch = confs_epoch[:crop_size]
        probs_epoch = probs_epoch[:crop_size]

        corrs = (preds_epoch == targets_epoch)

        if meter is not None:
            metric = meter.update(targets_epoch, preds_epoch, dataset)
        else:
            metric = -1

        if return_loss:
            mloss = torch.mean(torch.tensor(losses))
        else:
            mloss = 0

        return mloss, metric, corrs.cpu().numpy(), outputs_epoch, targets_epoch, probs_epoch.cpu().numpy(), confs_epoch.cpu().numpy(), preds_epoch.cpu().numpy(), sentences_per_second

    def run(self, model, dataloaders, optimizer, criterion, num_epochs=10, val_interval=5, meter=None):
        if meter is None:
            meter = Meter(self.config)
        tolerance = self.config['model']['tolerance']
        patience = self.config['model']['patience']
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        best_loss = torch.inf
        best_epoch = 0
        logging.info("Student model training")
        start_time = time.time()

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train(model, dataloaders["train"], optimizer, criterion)
            mlflow.log_metric("student/train/loss", train_loss)
            mlflow.log_metric("student/train/acc", train_acc)
            if epoch % val_interval == 0:
                val_loss, val_acc, corrs, outputs, targets, probs, _, _, _ = self.eval(model, dataloaders["val"], criterion, meter)

                mlflow.log_metric("student/val/loss", val_loss)
                mlflow.log_metric("student/val/acc", val_acc)
                

                if val_loss < best_loss * tolerance:
                    # perform temperature scaling and set the temperature in the model
                    self.ts.run(outputs, targets)
                    model.set_temperature(self.ts.temperature.item())

                    best_loss = val_loss
                    best_epoch = epoch
                    best_corrs = np.copy(corrs)
                    best_probs = (outputs/ self.ts.temperature.detach()).cpu().numpy()
                    torch.save(model.state_dict(), self.config['model']['model_weights_latest'])

                if epoch - best_epoch > patience * val_interval or corrs.mean() == 1:
                    break


        self.fitted = True
        logging.debug("total time: %0.2f (%d epochs)" % (time.time() - start_time, epoch))

        return best_loss, best_corrs, best_probs
        
    def init_student(self, dataset, config):
        logging.info("Initializing student model.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # the main moel uses semantic embeddings as input and predicts the category
        input_dim = dataset.embedding_size
        hidden_dim = np.ceil(dataset.n_classes * config['model']['hidden_size_factor']).astype(int)
        output_dim = dataset.n_classes
        model_main = MLP(input_dim, hidden_dim, output_dim)
        model_main = model_main.to(device)

        # only include those model parameters that require gradient updates
        parameters = filter(lambda p: p.requires_grad, model_main.parameters())
        
        optimizer_main = torch.optim.AdamW(parameters, lr=config['model']['learning_rate'], weight_decay=config['model']['weight_decay'], amsgrad=True)

        return model_main, optimizer_main
