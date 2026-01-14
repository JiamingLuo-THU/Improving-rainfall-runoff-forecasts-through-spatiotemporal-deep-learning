import os
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        self.patience    = patience
        self.verbose     = verbose
        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False
        self.val_loss_min= np.Inf

    def __call__(self, val_loss, model, optimizer, epoch, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_best(val_loss, model, optimizer, epoch, save_path)

        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement in {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            #if epoch % 50 == 0:
                #self._save_epoch(val_loss, model, optimizer, epoch, save_path)

        else:
            self.best_score = score
            self.counter = 0
            self._save_best(val_loss, model, optimizer, epoch, save_path)

    def _save_best(self, val_loss, model, optimizer, epoch, save_path):
        if self.verbose:
            print(f"[EarlyStopping] Validation loss decreased "
                  f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best model.")
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(ckpt, os.path.join(save_path, "best_model.pth.tar"))
        self.val_loss_min = val_loss

    def _save_epoch(self, val_loss, model, optimizer, epoch, save_path):
    
        if self.verbose:
            print(f"[EarlyStopping] Epoch {epoch}: saving checkpoint.")
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }
        fname = f"checkpoint_{epoch}_{val_loss:.6f}.pth.tar"
        torch.save(ckpt, os.path.join(save_path, fname))
