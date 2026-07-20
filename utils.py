import os
import random

import numpy as np
import torch
try:
    import dgl
except ImportError:
    dgl = None

def set_seed(seed):
    seed = int(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    # CUDA 确定性矩阵乘法配置。
    os.environ.setdefault(
        'CUBLAS_WORKSPACE_CONFIG',
        ':4096:8',
    )

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if dgl is not None:
        dgl.seed(seed)

        if hasattr(dgl, 'random'):
            dgl.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except TypeError:
        # 兼容不支持 warn_only 的旧版 PyTorch。
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """
    固定 DataLoader 子进程中的 NumPy 和 Python 随机性。
    """
    del worker_id

    worker_seed = (
        torch.initial_seed() % (2 ** 32)
    )
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x, eps=1e-12):

    x_min = x.min()
    x_max = x.max()
    denominator = x_max - x_min

    if torch.is_tensor(x):
        if denominator.abs().item() <= eps:
            return torch.zeros_like(x)
    else:
        if abs(float(denominator)) <= eps:
            return np.zeros_like(x)

    return (
        (x - x_min)
        / (denominator + eps)
    )

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x





