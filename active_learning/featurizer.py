import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

class Featurizer(object):
    def __init__(self, config, device=None):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'], num_labels=len(config['data']['label_dict']))
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Identity(in_features)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint'])

    def featurize(self, sample):
        
        self.model.to(self.device)

        tokens = self.tokenizer(sample, truncation=True)
        for key in tokens.keys():
            tokens[key] = torch.tensor(tokens[key]).unsqueeze(0)
            tokens[key] = tokens[key].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        return outputs.logits