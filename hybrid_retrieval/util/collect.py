import gc
import torch

def collect():
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()    