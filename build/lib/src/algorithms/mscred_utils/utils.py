import abc
from itertools import chain
import logging
import os
import pickle
import random
from typing import List, Union

import GPUtil
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal, lognorm, norm, chi
from tensorflow.python.client import device_lib
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange
import math

import ipdb

def fit_with_early_stopping(train_loader, val_loader, pytorch_module, patience, num_epochs, lr, verbose=True,
                            last_t_only=True, ret_best_val_loss=False):
    """
    :param train_loader: the pytorch data loader for the training set
    :param val_loader: the pytorch data loader for the validation set
    :param pytorch_module:
    :param patience:
    :param num_epochs: the maximum number of epochs for the training
    :param lr: the learning rate parameter used for optimization
    :return: trained module, avg train and val loss per epoch, final loss on train + val data per channel
    """
    pytorch_module.to(pytorch_module.device)  # .double()
    optimizer = torch.optim.Adam(pytorch_module.parameters(), lr=lr)
    epoch_wo_improv = 0
    pytorch_module.train()
    train_loss_by_epoch = []
    val_loss_by_epoch = []
    best_val_loss = None
    best_params = pytorch_module.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            if verbose:
                GPUtil.showUtilization()
            pytorch_module.train()
            train_loss = []
            for ts_batch in train_loader:
                ts_batch = ts_batch.float().to(pytorch_module.device)
                output = pytorch_module(ts_batch)
                assert last_t_only # output is reconstructed signature matrix at last time step
                loss = nn.MSELoss(reduction="mean")(output, ts_batch[:, :, -1, :, :])
                pytorch_module.zero_grad()
                loss.backward()
                optimizer.step()
                # multiplying by length of batch to correct accounting for incomplete batches
                train_loss.append(loss.item()*len(ts_batch))

            train_loss = np.mean(train_loss)/train_loader.batch_size
            train_loss_by_epoch.append(train_loss)

            # Get Validation loss
            pytorch_module.eval()
            val_loss = []
            with torch.no_grad():
                for ts_batch in val_loader:
                    ts_batch = ts_batch.float().to(pytorch_module.device)
                    output = pytorch_module(ts_batch)
                    assert last_t_only # output is reconstructed signature matrix at last time step
                    loss = nn.MSELoss(reduction="mean")(output, ts_batch[:, :, -1, :, :])
                    val_loss.append(loss.item()*len(ts_batch))

            val_loss = np.mean(val_loss)/val_loader.batch_size
            val_loss_by_epoch.append(val_loss)

            best_val_loss_epoch = np.argmin(val_loss_by_epoch)
            if best_val_loss_epoch == epoch:
                # any time a new best is encountered, the best_params will get replaced
                best_params = pytorch_module.state_dict()
                best_val_loss = val_loss
            # Check for early stopping by counting the number of epochs since val loss improved
            if epoch > 0 and val_loss >= val_loss_by_epoch[-2]:
                epoch_wo_improv += 1
            else:
                epoch_wo_improv = 0
        else:
            # early stopping is applied
            pytorch_module.load_state_dict(best_params)
            break
    pytorch_module.eval()
    val_reconstr_errors = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(pytorch_module.device)
            output = pytorch_module(ts_batch)
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, :, -1, :, :])
            val_reconstr_errors.append(error.cpu().numpy())
    if len(val_reconstr_errors) > 0:
        val_reconstr_errors = np.concatenate(val_reconstr_errors)
    if ret_best_val_loss:
        return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors, best_val_loss
    return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors


@torch.no_grad()
def predict_test_scores(pytorch_module, test_loader):
    pytorch_module.eval()
    reconstr_scores = []
    latent_points = []
    outputs_array = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(pytorch_module.device)
        output = pytorch_module(ts_batch)
        error = nn.L1Loss(reduction='none')(output, ts_batch[:, :, -1, :, :])
        error_dim = np.mean(error.cpu().numpy(), axis=(1,3))
        reconstr_scores.append(error_dim)
    reconstr_scores = np.concatenate(reconstr_scores)

    return reconstr_scores

