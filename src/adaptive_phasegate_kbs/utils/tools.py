import numpy as np
import torch
import os

def adjust_learning_rate(optimizer, epoch, args):
    # 根据用户深度建议：增加 Linear Warmup 并放慢衰减速度
    warmup_epochs = 5
    min_lr = 1e-7
    
    if epoch <= warmup_epochs:
        # 线性热身阶段
        lr = args.learning_rate * (epoch / warmup_epochs)
    elif args.lradj == 'type1':
        # 改进后的 type1：更温和的衰减 (Patience 调大，Factor 调为 0.8)
        # 前 15 个 Epoch 保持相对高位探索
        if epoch <= 15:
            lr = args.learning_rate
        else:
            # 之后每 4 个 Epoch 衰减 0.8
            lr = args.learning_rate * (0.8 ** ((epoch - 15) // 4))
    elif args.lradj == 'TST':
        # Cosine Annealing with Warmup
        lr = args.learning_rate * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    else:
        lr = args.learning_rate

    if lr is not None:
        if lr < min_lr: lr = min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:.10f}')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # 注意：如果 path 是 "./" 或 "."，不保存到磁盘（避免在根目录创建文件）
        # 实际最佳模型应该在训练脚本中保存到内存
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 只有当 path 不是当前目录时才保存到磁盘
        if path and path not in ["./", ".", ""]:
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean