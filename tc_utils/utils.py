import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_tokens(input_str):
    tokens = input_str.split(',')
    output_tokens = []
    for it in tokens:
        if it[0] == ' ':
            output_tokens.append(it[1:])
        else:
            output_tokens.append(it)

    return output_tokens
