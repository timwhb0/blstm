import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
class AudioDataset(Dataset):
    def __init__(self, inputs, labels1=None, labels2=None):
        self.inputs = inputs
        self.labels1 = labels1
        self.labels2 = labels2
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        if self.labels1 is not None and self.labels2 is not None:
            return self.inputs[index], self.labels1[index], self.labels2[index]
        else:
            return self.inputs[index]
        
def collate_fn(batch):
    inputs = [torch.tensor(item[0]) for item in batch]
    if len(batch[0]) == 3:
        labels1 = [torch.tensor(item[1]) for item in batch]
        labels2 = [torch.tensor(item[2]) for item in batch]
        return inputs, labels1, labels2
    else:
        return inputs,

def make_sequence(inputs, labels1=None, labels2=None):
    """
    Return a sequency for given inputs and corresponding labels (optional for test)
    Args:
        inputs: input feature vectors (i.e. magnitude of mixture speech)
        labels1: reference labels for target sepaker 1
        labels2: reference labels for target sepaker 2
    Returns:
        A list of tuples (input_sequence, label_sequence1, label_sequence2)
    """
    if labels1 is not None and labels2 is not None:
        return [(torch.tensor(inputs[i]), torch.tensor(labels1[i]), torch.tensor(labels2[i])) for i in range(len(inputs))]
    else:
        return [(torch.tensor(inputs[i]),) for i in range(len(inputs))]

