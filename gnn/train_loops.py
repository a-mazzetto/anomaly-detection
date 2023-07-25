"""Training loops"""
from typing import List
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score, average_precision_score
from cuda.cuda_utils import select_device

def graph_classification_train_loop(model, optimizer, loss_fn, num_epochs, train_loader, val_loader,
                                    metric, early_stopping, best_model_path=None, print_freq=None):
    """pytorch training loop. Deals with sending to device internally"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epoch_losses_train, epoch_acc_train = [], []
    epoch_losses_valid, epoch_acc_valid = [], []

    # Initialize metrics
    epoch_metric_train = metric
    epoch_metric_valid = deepcopy(metric)
    epoch_metric_train.to(device)
    epoch_metric_valid.to(device)
    
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

        model.eval()

        with torch.no_grad():

            # Remember that DataLoader shuffles the dataset at each iteration
            for batch in train_loader:
                batch = batch.to(device)
                predict = model(batch)
                loss = loss_fn(predict, batch.y)
                sum_loss_train += loss.item()
                _ = epoch_metric_train.update(predict.detach(), batch.y.detach())

            for batch in val_loader:
                batch = batch.to(device)

                predict = model(batch)
                loss = loss_fn(predict, batch.y)

                sum_loss_valid += loss.item()
                _ = epoch_metric_valid.update(predict.detach(), batch.y.detach())

        avg_epoch_loss_train = sum_loss_train / len(train_loader)
        avg_epoch_metric_train = epoch_metric_train.compute().cpu()

        avg_epoch_loss_valid = sum_loss_valid / len(val_loader)
        avg_epoch_metric_valid = epoch_metric_valid.compute().cpu()

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
    
    if best_model_path is not None:
        torch.save(best_model_state_dict, best_model_path)

    history = {
        'loss': epoch_losses_train,
        'val_loss': epoch_losses_valid,
        'accuracy': epoch_acc_train,
        'val_accuracy': epoch_acc_valid
    }

    return history

def graph_classification_train_loop(model, optimizer, loss_fn, num_epochs, train_loader, val_loader,
                                    metric, early_stopping, best_model_path=None, print_freq=None):
    """pytorch training loop. Deals with sending to device internally"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epoch_losses_train, epoch_acc_train = [], []
    epoch_losses_valid, epoch_acc_valid = [], []

    # Initialize metrics
    epoch_metric_train = metric
    epoch_metric_valid = deepcopy(metric)
    epoch_metric_train.to(device)
    epoch_metric_valid.to(device)
    
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

        model.eval()

        with torch.no_grad():

            # Remember that DataLoader shuffles the dataset at each iteration
            for batch in train_loader:
                batch = batch.to(device)
                predict = model(batch)
                loss = loss_fn(predict, batch.y)
                sum_loss_train += loss.item()
                _ = epoch_metric_train.update(predict.detach(), batch.y.detach())

            for batch in val_loader:
                batch = batch.to(device)

                predict = model(batch)
                loss = loss_fn(predict, batch.y)

                sum_loss_valid += loss.item()
                _ = epoch_metric_valid.update(predict.detach(), batch.y.detach())

        avg_epoch_loss_train = sum_loss_train / len(train_loader)
        avg_epoch_metric_train = epoch_metric_train.compute().cpu()

        avg_epoch_loss_valid = sum_loss_valid / len(val_loader)
        avg_epoch_metric_valid = epoch_metric_valid.compute().cpu()

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
    
    if best_model_path is not None:
        torch.save(best_model_state_dict, best_model_path)

    history = {
        'loss': epoch_losses_train,
        'val_loss': epoch_losses_valid,
        'accuracy': epoch_acc_train,
        'val_accuracy': epoch_acc_valid
    }

    return history

def multiple_roc_auc_score(x_true, x_pred):
    """List of matrices"""
    return np.array([roc_auc_score(_adj.numpy().flatten(), _pred.numpy().flatten()) for
                     _adj, _pred in zip(x_true, x_pred)]).mean()

def multiple_ap_score(x_true, x_pred):
    """List of matrices"""
    return np.array([average_precision_score(_adj.numpy().flatten(), _pred.numpy().flatten()) for
                     _adj, _pred in zip(x_true, x_pred)]).mean()

def vgrnn_train_loop(model, optimizer, num_epochs, train_loader, val_loader, early_stopping,
                     best_model_path=None, print_freq=None, device=None):
    """pytorch training loop for VGRNN. Deals with sending to device internally"""
    if device is None:
        device = select_device()
    model.to(device)

    epoch_losses_train, epoch_kls_train, epoch_nlls_train = [], [], []
    epoch_auc_train, epoch_ap_train = [], []
    epoch_losses_valid, epoch_kls_valid, epoch_nlls_valid = [], [], []
    epoch_auc_valid, epoch_ap_valid = [], []

    # Initialize metrics
    
    for epoch in range(num_epochs):
        # initialize train and validation loss to 0
        sum_loss_train, sum_kl_train, sum_nll_train = 0.0, 0.0, 0.0
        sum_auc_train, sum_ap_train = 0.0, 0.0
        sum_loss_valid, sum_kl_valid, sum_nll_valid = 0.0, 0.0, 0.0
        sum_auc_valid, sum_ap_valid = 0.0, 0.0

        model.train()

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            kl_loss, nll_loss, _, _, _, _ = model(
                torch.stack(unbatch(batch.x, batch=batch.batch)),
                unbatch_edge_index(batch.edge_index, batch=batch.batch),
                to_dense_adj(batch.edge_index, batch=batch.batch).to(device)
            )

            elbo = kl_loss + nll_loss

            elbo.backward()
            optimizer.step()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        model.eval()

        with torch.no_grad():

            # Remember that DataLoader shuffles the dataset at each iteration
            for batch in train_loader:
                batch = batch.to(device)
                dense_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                kl_loss, nll_loss, _, _, _, pred = model(
                    torch.stack(unbatch(batch.x, batch=batch.batch)),
                    unbatch_edge_index(batch.edge_index, batch=batch.batch),
                    dense_adj.to(device)
                )
                elbo = kl_loss + nll_loss
                sum_loss_train += elbo.item()
                sum_kl_train += kl_loss.item()
                sum_nll_train += nll_loss.item()
                sum_auc_train += multiple_roc_auc_score(dense_adj, pred.detach())
                sum_ap_train += multiple_ap_score(dense_adj, pred.detach())
                

            for batch in val_loader:
                batch = batch.to(device)
                dense_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                kl_loss, nll_loss, _, _, _, pred = model(
                    torch.stack(unbatch(batch.x, batch=batch.batch)),
                    unbatch_edge_index(batch.edge_index, batch=batch.batch),
                    dense_adj.to(device)
                )
                elbo = kl_loss + nll_loss
                sum_loss_valid += elbo.item()
                sum_kl_valid += kl_loss.item()
                sum_nll_valid += nll_loss.item()
                sum_auc_valid += multiple_roc_auc_score(dense_adj, pred.detach())
                sum_ap_valid += multiple_ap_score(dense_adj, pred.detach())

        avg_epoch_loss_train = sum_loss_train / len(train_loader)
        avg_epoch_kl_train = sum_kl_train / len(train_loader)
        avg_epoch_nll_train = sum_nll_train / len(train_loader)
        avg_epoch_auc_train = sum_auc_train / len(train_loader)
        avg_epoch_ap_train = sum_ap_train / len(train_loader)
        
        avg_epoch_loss_valid = sum_loss_valid / len(val_loader)
        avg_epoch_kl_valid = sum_kl_valid / len(val_loader)
        avg_epoch_nll_valid = sum_nll_valid / len(val_loader)
        avg_epoch_auc_valid = sum_auc_valid / len(val_loader)
        avg_epoch_ap_valid = sum_ap_valid / len(val_loader)

        if print_freq is not None and (epoch + 1) % print_freq == 0:
            print((f"Epoch {epoch + 1}: loss (kl + nll) - {avg_epoch_loss_train:.4f} "
                   f"({avg_epoch_kl_train:.4f} + {avg_epoch_nll_train:.4f}), "
                   f"val_loss (kl + nll) - {avg_epoch_loss_valid:.4f}, "
                   f"({avg_epoch_kl_valid:.4f} + {avg_epoch_nll_valid:.4f})"))

        epoch_losses_train.append(avg_epoch_loss_train)
        epoch_kls_train.append(avg_epoch_kl_train)
        epoch_nlls_train.append(avg_epoch_nll_train)
        epoch_auc_train.append(avg_epoch_auc_train)
        epoch_ap_train.append(avg_epoch_ap_train)
        
        epoch_losses_valid.append(avg_epoch_loss_valid)
        epoch_kls_valid.append(avg_epoch_kl_valid)
        epoch_nlls_valid.append(avg_epoch_nll_valid)
        epoch_auc_valid.append(avg_epoch_auc_valid)
        epoch_ap_valid.append(avg_epoch_ap_valid)

        if early_stopping.min_valid_loss >= avg_epoch_loss_valid:
            best_model_state_dict = deepcopy(model.state_dict())

        if early_stopping.early_stop(avg_epoch_loss_valid):
            break
    
    if best_model_path is not None:
        torch.save(best_model_state_dict, best_model_path)

    history = {
        'loss': epoch_losses_train,
        'kl': epoch_kls_train,
        'nll': epoch_nlls_train,
        'auc': epoch_auc_train,
        'ap': epoch_ap_train,

        'val_loss': epoch_losses_valid,
        'val_kl': epoch_kls_valid,
        'val_nll': epoch_nlls_valid,
        'val_auc': epoch_auc_valid,
        'val_ap': epoch_ap_valid,
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
        ax[0, 1].plot(moving_average(history['accuracy'], 10), color=COLORS[num], label=names[num])

        ax[1, 0].plot(history['val_loss'], alpha=0.2, color=COLORS[num])
        ax[1, 0].plot(moving_average(history['val_loss'], 10), color=COLORS[num], label=names[num])

        ax[1, 1].plot(history['val_accuracy'], alpha=0.2, color=COLORS[num])
        ax[1, 1].plot(moving_average(history['val_accuracy'], 10), color=COLORS[num], label=names[num])

    _ = [axis.legend() for axis in ax.flatten()]

    return fig

def plot_vgrnn_training_result(history: List[dict]):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    COLORS = ["blue", "red", "green", "cyan", "magenta"]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()
    ax[0, 0].set_title('Loss')
    ax[0, 1].set_title('Loss valid')
    ax[1, 0].set_title('Metrics')
    ax[1, 1].set_title('Metrics valid')
    _ = [axis.grid() for axis in ax.flatten()]

    ax[0, 0].plot(history['loss'], alpha=0.2, color=COLORS[0])
    ax[0, 0].plot(history['kl'], alpha=0.2, color=COLORS[1])
    ax[0, 0].plot(history['nll'], alpha=0.2, color=COLORS[2])
    ax[0, 0].plot(moving_average(history['loss'], 10), color=COLORS[0], label="ELBO")
    ax[0, 0].plot(moving_average(history['kl'], 10), color=COLORS[1], label="KL")
    ax[0, 0].plot(moving_average(history['nll'], 10), color=COLORS[2], label="NLL")

    ax[0, 1].plot(history['val_loss'], alpha=0.2, color=COLORS[0])
    ax[0, 1].plot(history['val_kl'], alpha=0.2, color=COLORS[1])
    ax[0, 1].plot(history['val_nll'], alpha=0.2, color=COLORS[2])
    ax[0, 1].plot(moving_average(history['val_loss'], 10), color=COLORS[0], label="ELBO")
    ax[0, 1].plot(moving_average(history['val_kl'], 10), color=COLORS[1], label="KL")
    ax[0, 1].plot(moving_average(history['val_nll'], 10), color=COLORS[2], label="NLL")

    ax[1, 0].plot(history['auc'], alpha=0.2, color=COLORS[0])
    ax[1, 0].plot(history['ap'], alpha=0.2, color=COLORS[1])
    ax[1, 0].plot(moving_average(history['auc'], 10), color=COLORS[0], label="AUC")
    ax[1, 0].plot(moving_average(history['ap'], 10), color=COLORS[1], label="AP")

    ax[1, 1].plot(history['val_auc'], alpha=0.2, color=COLORS[0])
    ax[1, 1].plot(history['val_ap'], alpha=0.2, color=COLORS[1])
    ax[1, 1].plot(moving_average(history['val_auc'], 10), color=COLORS[0], label="AUC")
    ax[1, 1].plot(moving_average(history['val_ap'], 10), color=COLORS[1], label="AP")

    _ = [axis.legend() for axis in ax.flatten()]

    return fig
