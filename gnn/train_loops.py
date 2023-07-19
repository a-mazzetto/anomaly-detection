"""Training loops"""
from typing import List
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader

def graph_classification_train_loop(model, optimizer, loss_fn, num_epochs, train_loader, val_loader,
                                    metric, early_stopping, best_model_path, print_freq=None):
    """pytorch training loop. Deals with sending to device internally"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epoch_losses_train, epoch_acc_train = [], []
    epoch_losses_valid, epoch_acc_valid = [], []

    # Initialize metrics
    epoch_metric_train = metric
    epoch_metric_valid = deepcopy(metric)
    
    for epoch in range(num_epochs):
        # initialize train and validation loss to 0
        sum_loss_train = 0.0
        sum_loss_valid = 0.0
        _ = epoch_metric_train.reset()
        _ = epoch_metric_valid.reset()

        model.train()

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            predict = model(batch)

            loss = loss_fn(predict, batch.y)

            loss.backward()
            optimizer.step()

            sum_loss_train += loss.item()
            # Use detach to avoid affecting gradients
            _ = epoch_metric_train.update(predict.detach(), batch.y.detach())

        model.eval()

        with torch.no_grad():

            for batch in val_loader:
                batch = batch.to(device)

                predict = model(batch)
                loss = loss_fn(predict, batch.y)

                sum_loss_valid += loss.item()
                _ = epoch_metric_valid.update(predict.detach(), batch.y.detach())

        avg_epoch_loss_train = sum_loss_train / len(train_loader)
        avg_epoch_metric_train = epoch_metric_train.compute()

        avg_epoch_loss_valid = sum_loss_valid / len(val_loader)
        avg_epoch_metric_valid = epoch_metric_valid.compute()

        if print_freq is not None and (epoch + 1) % print_freq == 0:
            print((f"Epoch {epoch + 1}: loss - {avg_epoch_loss_train:.4f}, metric = {avg_epoch_metric_train:.4f}, "
                   f"val_loss - {avg_epoch_loss_valid:.4f}, val_metric = {avg_epoch_metric_valid:.4f}"))

        epoch_losses_train.append(avg_epoch_loss_train)
        epoch_acc_train.append(avg_epoch_metric_train)

        epoch_losses_valid.append(avg_epoch_loss_valid)
        epoch_acc_valid.append(avg_epoch_metric_valid)

        if early_stopping.min_valid_loss >= avg_epoch_loss_valid:
            best_model_state_dict = deepcopy(model.state_dict())

        if early_stopping.early_stop(avg_epoch_loss_valid):
            break

    torch.save(best_model_state_dict, best_model_path)

    history = {
        'loss': epoch_losses_train,
        'val_loss': epoch_losses_valid,
        'accuracy': epoch_acc_train,
        'val_accuracy': epoch_acc_valid
    }

    return history

# Plotting
def plot_training_results(histories: List[dict], names=None):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    COLORS = ["blue", "red", "green", "cyan", "magenta"]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()
    ax[0, 0].set_title('Train loss')
    ax[0, 1].set_title('Train accuracy')
    ax[1, 0].set_title('Validation loss')
    ax[1, 1].set_title('Validation accuracy')
    _ = [axis.grid() for axis in ax.flatten()]

    for num, history in enumerate(histories):
        ax[0, 0].plot(history['loss'], alpha=0.2, color=COLORS[num])
        ax[0, 0].plot(moving_average(history['loss'], 10), color=COLORS[num], label=names[num])

        ax[0, 1].plot(history['accuracy'], alpha=0.2, color=COLORS[num])
        ax[0, 1].plot(moving_average(history['loss'], 10), color=COLORS[num], label=names[num])

        ax[1, 0].plot(history['val_loss'], alpha=0.2, color=COLORS[num])
        ax[1, 0].plot(moving_average(history['loss'], 10), color=COLORS[num], label=names[num])

        ax[1, 1].plot(history['val_accuracy'], alpha=0.2, color=COLORS[num])
        ax[1, 1].plot(moving_average(history['loss'], 10), color=COLORS[num], label=names[num])

    _ = [axis.legend() for axis in ax.flatten()]

    return fig
