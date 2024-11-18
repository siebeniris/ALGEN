import torch


def get_weights_from_attention_mask(attention_mask):
    weights = attention_mask / attention_mask.sum(dim=1, keepdim=True)
    return weights


def get_device():
    """Determine the best available device for macOS"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
