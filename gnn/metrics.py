"""Custom Metrics"""
import torch
from torchmetrics import Metric

class CrossEntropyAccuracyFromLogits(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.tensor):
        predicted_class = torch.nn.functional.softmax(preds, dim=0).argmax(dim=1)
        real_class = target

        self.correct += torch.sum(predicted_class == real_class)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
