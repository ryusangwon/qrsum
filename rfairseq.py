import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained('/path/to/bart.large', checkpoint_file='model.pt')
bart = torch.hub.load("pytorch/fairseq", args.model_file)