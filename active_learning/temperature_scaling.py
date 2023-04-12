import json
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        accs_binned = []
        confs_binned = []

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                confs_binned.append(avg_confidence_in_bin.item())
                accs_binned.append(accuracy_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            else:
                confs_binned.append(((bin_lower + bin_upper) / 2).item())
                accs_binned.append(0)

        return ece, confs_binned, accs_binned


class TemperatureScaling():
    def __init__(self, config, temperature=2, nbins=15):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config

        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.nbins = nbins
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam([self.temperature])

    def run(self, logits, labels):
        ece_criterion = ECELoss(self.nbins).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        # try:
        before_temperature_nll = self.criterion(logits, labels).item()
        # except:
        #     import rpdb
        #     rpdb.set_trace()
        before_temperature_ece = ece_criterion(logits, labels)
        print('Before - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece[0].item()))

        self.search(logits, labels)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = self.criterion(self.scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.scale(logits), labels)
        print('After - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece[0].item()))
        print('Optimal temperature: %.3f (%d epochs)' % (self.temperature.item(), self.epochs))

        self.plot(before_temperature_ece, after_temperature_ece)

        # return self.temperature.item(), before_temperature_ece, after_temperature_ece
    
    def plot(self, before, after):
        # ece_c, accs_c, bins_c = calibrate(temp)
        # ece, accs, bins = calibrate(1)
        import seaborn as sns
        import matplotlib.pyplot as plt

        filename = self.config['misc']['output_path'] + os.path.sep + 'temperature_scaling.png'

        sns.set_style("whitegrid")
        plt.plot(after[1], after[2], "x-", color='green', label='calibrated')
        plt.plot(before[1], before[2], "x-", color='orange', label='raw')
        plt.plot((0,1), (0,1), "--")
        plt.xlabel('confidence')
        plt.ylabel('accuracy')
        plt.title('temp: %0.2f, ece: %0.2f (w/o scaling %0.2f)' % (self.temperature.item(), after[0], before[0]))
        _ = plt.legend()
        plt.savefig(filename)
        plt.close()

    def scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def search(self, logits, labels, max_epoch=10000):

        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        best_loss = np.inf
        best_epoch = 0
        patience = 100
        
        for epoch in range(max_epoch):
            self.optimizer.zero_grad()
            loss = self.criterion(self.scale(logits), labels)
            if loss < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
            
            if epoch - best_epoch > patience:
                break
            loss.backward()
            self.optimizer.step()
        self.epochs = epoch
