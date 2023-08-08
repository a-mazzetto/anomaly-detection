"""Custom Metrics"""
import torch
from torchmetrics import Metric

class CrossEntropyAccuracyFromLogits(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.tensor):
        predicted_class = preds.argmax(dim=1)

        self.correct += torch.sum(predicted_class == target).int()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

# Custom losses
def roc_auc_loss(labels, logits, weights=1.0):
    """https://github.com/tensorflow/models/blob/archive/research/global_objectives/loss_layers.py"""
    # Input processing
    assert labels.shape == logits.shape, "Same shape expected"
    if isinstance(weights, (float, int)):
        weights = weights * torch.ones_like(labels)
    assert labels.shape == weights.shape, "Unexpected weights shape"
    original_shape = labels.shape
    labels = labels.flatten().type(torch.float)
    logits = logits.flatten().type(torch.float)
    weights = weights.flatten().type(torch.float)

    labels_diff = torch.unsqueeze(labels, dim=0) - torch.unsqueeze(labels, dim=1)
    logits_diff = torch.unsqueeze(logits, dim=0) - torch.unsqueeze(logits, dim=1)
    weights_prod = torch.unsqueeze(weights, dim=0) * torch.unsqueeze(weights, dim=1)

    signed_logits_diff = labels_diff * logits_diff

    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.ones_like(signed_logits_diff),
        signed_logits_diff,
        reduction="none"
    )

    weighted_loss = weights_prod * raw_loss

    loss = torch.mean(torch.abs(labels_diff) * weighted_loss, dim=[0]) * 0.5
    loss = torch.reshape(loss, original_shape)

    return loss
