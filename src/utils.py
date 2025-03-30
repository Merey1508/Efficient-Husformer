import torch
import os
from src.dataset import Multimodal_Datasets
import torch.nn as nn

def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, weights_only=False)
    return data


def save_load_name(args, name=''):
    load_name = name + '_' + args.model
    return load_name


def save_model(args, model, name=''):
    if not os.path.exists('output/'):
        os.makedirs('output/')
    name = save_load_name(args, name)
    torch.save(model, f'output/{args.name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'output/{args.name}.pt')
    return model

def remake_label(target):
    """
    Ensure that target labels are in the correct format for focal loss computation.
    This function can be modified based on specific dataset requirements.
    """
    if target.dim() > 1:  
        target = target.argmax(dim=1)  # Convert one-hot to class indices if necessary
    return target


class focalloss(nn.Module):
    def __init__(self, alpha=[0.4,0.3,0.3], gamma=2, reduction='mean'):
        super(focalloss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        target = remake_label(target).type(torch.int64)

        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt 
        pt = torch.exp(logpt)
        focal_loss = alpha * ((1 - pt) ** self.gamma * ce_loss).t()
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
