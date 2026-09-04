import gc
import torch


def collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
