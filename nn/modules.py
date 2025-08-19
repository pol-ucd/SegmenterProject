import numpy as np
import torch
import torch.nn as nn

from losses import TverskyLoss as TL, FocalLoss as FL, DiceLoss, JaccardLoss

"""
Implements Hanija's Combined Loss for Binary image classification.
Is also callable so it can be used to evaluate loss with no_grad()
"""


class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        if weights is None:
            weights = {'bce': 0.2, 'tversky': 0.4, 'focal': 0.4, 'dice': 0.6, 'jaccard': 0.6}
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TL(alpha=0.2, beta=0.4, mode='binary')
        self.focal = FL(mode='binary')
        self.dice = DiceLoss(mode='binary')
        self.iou = JaccardLoss(mode='binary')

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        bce = self.bce(pred, target.unsqueeze(1).float())
        pred = pred.transpose(3, 1)
        tversky = self.tversky(pred, target.float())
        focal = self.focal(pred, target.float())
        dice = self.dice(pred, target.float())
        jaccard = self.iou(pred, target.float())
        return (self.weights['bce'] * bce + self.weights['tversky'] * tversky
                + self.weights['focal'] * focal + self.weights['dice'] * dice
                + self.weights['jaccard'] * jaccard)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=False, save_path=None):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy or IoU.
            verbose (bool): If True, prints updates.
            save_path (str): If set, saves best model to this path.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = None

        self._init_comparator()

    def _init_comparator(self):
        if self.mode == 'min':
            self.compare = lambda current, best: current < best - self.min_delta
            self.best_score = np.inf
        else:
            self.compare = lambda current, best: current > best + self.min_delta
            self.best_score = -np.inf

    def __call__(self, current_score, model=None, epoch=None):
        if self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"New best score: {current_score:.4f} at epoch {epoch}")
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"Model saved to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = None
