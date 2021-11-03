import torch


def ifNan(model):
    for key, value in model.state_dict().items():
        if torch.any(torch.isnan(value)):
            print(key)
