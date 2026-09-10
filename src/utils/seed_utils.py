import random, torch, numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ggf. noch torch.backends.cudnn.deterministic = True / benchmark = False
