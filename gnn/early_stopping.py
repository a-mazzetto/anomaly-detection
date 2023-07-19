"""Early stopping"""
import numpy as np

class EarlyStopping:
    
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.min_valid_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_valid_loss:
            self.min_valid_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_valid_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
