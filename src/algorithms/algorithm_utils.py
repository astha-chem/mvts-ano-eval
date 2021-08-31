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


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False, out_dir=None):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}
        self.out_dir = out_dir
        self.torch_save = False

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """

    def set_output_dir(self, out_dir):
        self.out_dir = out_dir

    def get_val_err(self):
        """
        :return: reconstruction error_tc for validation set,
        dimensions of num_val_time_points x num_channels
        Call after training
        """
        return None

    def get_val_loss(self):
        """
        :return: scalar loss after training
        """
        return None


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0
        self.torch_save = True

    @property
    def device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class TensorflowUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            tf.set_random_seed(seed)
        self.framework = 1

    @property
    def device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return tf.device(gpus[self.gpu] if gpus and self.gpu is not None else '/cpu:0')


# class AedUtils(metaclass=abc.ABCMeta):
#     def __init__(self):
#         pass
#
#     @staticmethod
def get_sub_seqs(x_arr, seq_len, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs


def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits: List, seed: int, shuffle: bool = False,
    usetorch = True):
    """
    Splits the train data between train, val, etc. Creates and returns pytorch data loaders
    :param shuffle: boolean that determines whether samples are shuffled before splitting the data
    :param seed: seed used for the random shuffling (if shuffling there is)
    :param x_seqs: input data where each row is a sample (a sequence) and each column is a channel
    :param batch_size: number of samples per batch
    :param splits: list of split fractions, should sum up to 1.
    :param usetorch: if True returns dataloaders, otherwise return datasets
    :return: a tuple of data loaders as long as splits. If len_splits = 1, only 1 data loader is returned
    """
    if np.sum(splits) != 1:
        scale_factor = np.sum(splits)
        splits = [fraction/scale_factor for fraction in splits]
    if shuffle:
        np.random.seed(seed)
        x_seqs = x_seqs[np.random.permutation(len(x_seqs))]
        np.random.seed()
    split_points = [0]
    for i in range(len(splits)-1):
        split_points.append(split_points[-1] + int(splits[i]*len(x_seqs)))
    split_points.append(len(x_seqs))
    if usetorch:
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]], batch_size=batch_size,
            drop_last=False, pin_memory=True, shuffle=False) for i in range(len(splits))])
        return loaders
    else:
        # datasets = tuple([x_seqs[split_points[i]: 
        #     (split_points[i] + (split_points[i+1]-split_points[i])//batch_size*batch_size)] 
        #     for i in range(len(splits))])
        datasets = tuple([x_seqs[split_points[i]:split_points[i+1]]
            for i in range(len(splits))])
        return datasets


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
                if last_t_only:
                    loss = nn.MSELoss(reduction="mean")(output[:, -1], ts_batch[:, -1])
                else:
                    loss = nn.MSELoss(reduction="mean")(output, ts_batch)
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
                    if last_t_only:
                        loss = nn.MSELoss(reduction="mean")(output[:, -1], ts_batch[:, -1])
                    else:
                        loss = nn.MSELoss(reduction="mean")(output, ts_batch)
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
            output = pytorch_module(ts_batch)[:, -1]
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            val_reconstr_errors.append(error.cpu().numpy())
    if len(val_reconstr_errors) > 0:
        val_reconstr_errors = np.concatenate(val_reconstr_errors)
    if ret_best_val_loss:
        return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors, best_val_loss
    return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors


def predict_univar_outputs_encodings(trained_univar_model, ts_batch):
    univar_outputs = []
    univar_encodings = []
    for channel_num, univar_model in enumerate(trained_univar_model):
        univar_output, univar_encoding = univar_model(ts_batch[:, :, channel_num].unsqueeze(2), return_latent=True)
        if len(univar_encoding.shape) == 2:
            univar_encoding = univar_encoding.unsqueeze(1)
        univar_outputs.append(univar_output[:, -1])  # output shape is [batch_size, 1], encoded shape is [batch_size, 5]
        univar_encodings.append(univar_encoding)
    univar_outputs = torch.cat(univar_outputs, dim=1)
    univar_errors = ts_batch[:, -1, :] - univar_outputs
    univar_encodings = torch.stack(univar_encodings).permute(1, 0, 2, 3)
    return univar_errors, univar_outputs, univar_encodings

def fit_with_early_stopping_residual_joint(train_loader, val_loader,
                                           untrained_univar_model: List[nn.Module],
                                           pytorch_module, patience, num_epochs, lr, verbose=True):
    """
    :param train_loader: the pytorch data loader for the training set
    :param val_loader: the pytorch data loader for the validation set
    :param untrained_univar_model: untrained model, may already set into eval mode. Assumed to be pytorch model.
    :param pytorch_module:
    :param patience:
    :param num_epochs: the maximum number of epochs for the training
    :param lr: the learning rate parameter used for optimization
    :return: trained module, avg train and val loss per epoch, final loss on train + val data per channel
    """
    pytorch_module.to(pytorch_module.device)  # .double()
    [model.to(pytorch_module.device) for model in untrained_univar_model]
    all_params = [model.parameters() for model in untrained_univar_model] + [pytorch_module.parameters()]
    optimizer = torch.optim.Adam(chain(*all_params), lr=lr)
    epoch_wo_improv = 0
    train_loss_by_epoch = []
    val_loss_by_epoch = []
    for epoch in trange(num_epochs):
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            if verbose:
                GPUtil.showUtilization()
            pytorch_module.train()
            [model.train() for model in untrained_univar_model]
            train_loss = []
            for ts_batch in train_loader:
                ts_batch = ts_batch.float().to(pytorch_module.device)
                univar_errors, univar_outputs, univar_encodings = predict_univar_outputs_encodings(untrained_univar_model, ts_batch)
                output = pytorch_module(univar_encodings)
                loss = nn.MSELoss(reduction="mean")(output, univar_errors) + nn.MSELoss(reduction="mean")(univar_outputs, ts_batch[:, -1])
                pytorch_module.zero_grad()
                [model.zero_grad() for model in untrained_univar_model]
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            train_loss_by_epoch.append(train_loss)

            # Get Validation loss
            pytorch_module.eval()
            [model.eval() for model in untrained_univar_model]
            val_loss = []
            with torch.no_grad():
                for ts_batch in val_loader:
                    ts_batch = ts_batch.float().to(pytorch_module.device)
                    with torch.no_grad():
                        univar_errors, _, univar_encodings = predict_univar_outputs_encodings(untrained_univar_model,
                                                                                              ts_batch)
                    output = pytorch_module(univar_encodings)
                    loss = nn.MSELoss(reduction="mean")(output, univar_errors) + nn.MSELoss(reduction="mean")(univar_outputs, ts_batch[:, -1])
                    val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            val_loss_by_epoch.append(val_loss)
            # Check for early stopping by counting the number of epochs since val loss improved
            if epoch > 1:
                if val_loss_by_epoch[-1] >= val_loss_by_epoch[-2]:
                    epoch_wo_improv += 1
                    if epoch_wo_improv == 1:
                        before_overfit_par = pytorch_module.state_dict()
                else:
                    epoch_wo_improv = 0
        else:
            pytorch_module.load_state_dict(before_overfit_par)
            break
    pytorch_module.eval()
    [model.eval() for model in untrained_univar_model]
    val_reconstr_errors = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(pytorch_module.device)
            with torch.no_grad():
                _, univar_outputs, univar_encodings = predict_univar_outputs_encodings(untrained_univar_model,
                                                                                       ts_batch)
            output = univar_outputs + pytorch_module(univar_encodings)
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            val_reconstr_errors.append(torch.squeeze(error).cpu().numpy())
    val_reconstr_errors = np.concatenate(val_reconstr_errors)
    return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors


def fit_with_early_stopping_residual(train_loader, val_loader,
                                     trained_univar_model: List[nn.Module],
                                     pytorch_module, patience, num_epochs, lr, verbose=True):
    """
    :param train_loader: the pytorch data loader for the training set
    :param val_loader: the pytorch data loader for the validation set
    :param trained_univar_model: pre-trained model, already set into eval mode. Assumed to be pytorch model.
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
    for epoch in trange(num_epochs):
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            if verbose:
                GPUtil.showUtilization()
            pytorch_module.train()
            train_loss = []
            for ts_batch in train_loader:
                ts_batch = ts_batch.float().to(pytorch_module.device)
                with torch.no_grad():
                    univar_errors, _, univar_encodings = predict_univar_outputs_encodings(trained_univar_model, ts_batch)
                output = pytorch_module(univar_encodings)
                loss = nn.MSELoss(reduction="mean")(output, univar_errors)
                pytorch_module.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            train_loss_by_epoch.append(train_loss)

            # Get Validation loss
            pytorch_module.eval()
            val_loss = []
            with torch.no_grad():
                for ts_batch in val_loader:
                    ts_batch = ts_batch.float().to(pytorch_module.device)
                    with torch.no_grad():
                        univar_errors, _, univar_encodings = predict_univar_outputs_encodings(trained_univar_model,
                                                                                              ts_batch)
                    output = pytorch_module(univar_encodings)
                    loss = nn.MSELoss(reduction="mean")(output, univar_errors)
                    val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            val_loss_by_epoch.append(val_loss)
            # Check for early stopping by counting the number of epochs since val loss improved
            if epoch > 1:
                if val_loss_by_epoch[-1] >= val_loss_by_epoch[-2]:
                    epoch_wo_improv += 1
                    if epoch_wo_improv == 1:
                        before_overfit_par = pytorch_module.state_dict()
                else:
                    epoch_wo_improv = 0
        else:
            pytorch_module.load_state_dict(before_overfit_par)
            break
    pytorch_module.eval()
    val_reconstr_errors = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(pytorch_module.device)
            with torch.no_grad():
                _, univar_outputs, univar_encodings = predict_univar_outputs_encodings(trained_univar_model,
                                                                                      ts_batch)
            output = univar_outputs + pytorch_module(univar_encodings)
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            val_reconstr_errors.append(torch.squeeze(error).cpu().numpy())
    val_reconstr_errors = np.concatenate(val_reconstr_errors)
    return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors


@torch.no_grad()
def predict_test_scores(pytorch_module, test_loader, latent=False, return_output=False):
    pytorch_module.eval()
    reconstr_scores = []
    latent_points = []
    outputs_array = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(pytorch_module.device)
        if latent:
            output, encoding = pytorch_module(ts_batch, return_latent=latent)
            output = output[:, -1]
            latent_points.append(torch.squeeze(encoding).cpu().numpy())
        else:
            output = pytorch_module(ts_batch)[:, -1]
        error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
        reconstr_scores.append(error.cpu().numpy())
        outputs_array.append(output.cpu().numpy())
    reconstr_scores = np.concatenate(reconstr_scores)
    outputs_array = np.concatenate(outputs_array)
    if latent:
        latent_points = np.concatenate(latent_points)
        padding = np.zeros((len(ts_batch[0]) - 1, latent_points.shape[1]))
        latent_points = np.concatenate([padding, latent_points])
    multivar = (len(reconstr_scores.shape) > 1)
    if multivar:
        padding = np.zeros((len(ts_batch[0]) - 1, reconstr_scores.shape[-1]))
    else:
        padding = np.zeros(len(ts_batch[0]) - 1)
    reconstr_scores = np.concatenate([padding, reconstr_scores])
    outputs_array = np.concatenate([padding, outputs_array])

    if latent and return_output:
        return_vars = (reconstr_scores, latent_points, outputs_array)
    elif latent:
        return_vars = (reconstr_scores, latent_points)
    elif return_output:
        return_vars = (reconstr_scores, outputs_array)
    else:
        return_vars = reconstr_scores

    return return_vars


def combine_multiple_preds_per_time(scores, stride, num_time_points, method="ewma", ewma_decay=0.9):
    """ To enforce causality we only use the prediction of the last time sep in each subsequence so this method is not
    used anymore
    :param method: alternative methods might be average, lin_reg
    :param ewma_decay: decay parameter for the ewma method
    :param num_time_points: not sure if this is absolutely needed
    :param stride: number of time points the sequencing window has moved between two subsequences during
    preprocessing
    :param scores: anomaly scores 3d array, dim 0 is the samples (subsequences), dim 1 is sequence length, dim 3 is
    channels
    :return: 2d array containing ensembled scores with dim 0 being num_time_points and dim 1 being the channels
    """
    num_sequences = scores.shape[0]
    seq_len = scores.shape[1]
    num_channels = scores.shape[2]
    lattice = np.full((seq_len, num_time_points, num_channels), np.nan)
    combined_scores = np.zeros((num_time_points, num_channels))
    for i in range(seq_len):
        lattice[i, np.arange(i, i + num_sequences * stride, stride), :] = scores[:, i, :]
    if method == "ewma":
        # The current implementation for ewma is probably computationally not that good
        for k in range(num_time_points):
            non_nan_scores = lattice[~np.isnan(lattice[:, k, 0]), k, :]
            if non_nan_scores.shape[0] > 0:
                ewma = non_nan_scores[0, :]
            else:
                ewma = np.zeros(num_channels)
            for j in range(1, non_nan_scores.shape[0]):
                ewma = (1 - ewma_decay)*ewma + ewma_decay*non_nan_scores[j, :]
            combined_scores[k, :] = ewma
    elif method == 'average':
        combined_scores = np.nanmean(lattice, axis=0)
    else:
        print("This method of combining scores for the same time point but from different sub sequences is not \
        implemented, averaging will be used")
        combined_scores = np.nanmean(lattice, axis=0)
    return combined_scores


def fit_scores_distribution(scores_arr: np.ndarray, distr='multivar_gaussian'):
    """
    :param scores_arr: 2d array where dim 0 is time, dim 1 is channels. The scores have already been combined so rows
    correspond to distinct time points
    :param distr: the name of the distribution to be fitted to anomaly scores on train data
    :return: params dict with distr name and parameters of distribution
    """
    distr_parameters = {"distr": distr}
    if distr == 'multivar_gaussian':
        mean = np.mean(scores_arr, axis=0)
        cov = np.cov(scores_arr, rowvar=False)
        distr_parameters["mean"] = mean
        distr_parameters["covariance"] = cov
    elif distr == 'univar_gaussian':
        means = np.mean(scores_arr, axis=0)
        var = np.var(scores_arr, axis=0)
        distr_parameters["mean"] = means
        distr_parameters["variance"] = var
    elif distr == 'univar_lognormal':
        shapes, locs, scales = [], [], []
        for channel in range(scores_arr.shape[1]):
            shape, loc, scale = lognorm.fit(scores_arr[:, channel])
            shapes.append(shape)
            locs.append(loc)
            scales.append(scale)
        distr_parameters["shape"] = np.array(shapes)
        distr_parameters["loc"] = np.array(locs)
        distr_parameters["scale"] = np.array(scales)
    elif distr == 'multivar_lognormal':
        log_scores = np.log(scores_arr)
        normal_mean = np.mean(log_scores, axis=0)
        normal_cov = np.cov(log_scores, rowvar=False)
        distr_parameters["normal_mean"] = normal_mean
        distr_parameters["normal_covariance"] = normal_cov
    elif distr == "chi":
        dfs = []
        for channel in range(scores_arr.shape[1]):
            estimated_df = chi.fit(scores_arr[:, channel].ravel())[0]
            df = round(estimated_df)
            dfs.append(df)
        distr_parameters["df"] = dfs
    else:
        print("This distribution is unknown or has not been implemented yet, a multivariate gaussian distribution \
        will be fitted to the anomaly scores on the train set")
        mean = np.mean(scores_arr, axis=0)
        cov = np.cov(scores_arr, rowvar=False)
        distr_parameters["mean"] = mean
        distr_parameters["covariance"] = cov
    return distr_parameters


def get_logp_from_dist(pred_scores_arr: np.ndarray, params: dict, neg=True, log=True):
    """
    :param pred_scores_arr: 2d array where dim 0 is samples and dim 1 is channels, there can be only 1 channel,
    but it should still be a 2D array, the scores have already been combined so rows are time points
    :param params: must contain key 'distr' and corresponding params
    :param neg: True results in anomaly score
    :return: probabilities or log probabilities of length same as pred_scores_arr.shape[0]
    """
    distr = params["distr"]
    sign = -(2 * int(neg) - 1)
    if distr == 'multivar_gaussian':
        assert ("mean" in params.keys() and "covariance" in params.keys()), "The mean and/or covariance are missing, \
        we can't define the distribution"
        distribution = multivariate_normal(mean=params["mean"], cov=params["covariance"], allow_singular=True)
    elif distr == 'univar_gaussian':
        assert ("mean" in params.keys() and "variance" in params.keys()), "The mean and/or variance are missing, \
        we can't define the distribution"
        # We use here a multivariate gaussian but where the channels are independant from each other
        distribution = multivariate_normal(mean=params["mean"], cov=np.diag(params["variance"]), allow_singular=True)
    else:
        print("This distribution is unknown or has not been implemented yet, a multivariate gaussian will be used")
        assert ("mean" in params.keys() and "covariance" in params.keys()), "The mean or covariance are missing, \
        we can't define a gaussian distribution"
        distribution = multivariate_normal(mean=params["mean"], cov=params["covariance"], allow_singular=True)
    if log:
        probability_scores = sign*distribution.logpdf(pred_scores_arr)
    else:
        probability_scores = sign * distribution.pdf(pred_scores_arr)
    return probability_scores


def get_per_channel_probas(pred_scores_arr, params, logcdf=False):
    """
    :param pred_scores_arr: 1d array of the reconstruction errors for one channel
    :param params: must contain key 'distr' and corresponding params
    :return: array of negative log pdf of same length as pred_scores_arr
    """
    distr = params["distr"]
    probas = None
    constant_std = 0.000001
    if distr == "univar_gaussian":
        assert ("mean" in params.keys() and ("std" in params.keys()) or "variance" in params.keys()), \
            "The mean and/or standard deviation are missing, we can't define the distribution"
        if "std" in params.keys():
            if params["std"] == 0.0:
                params["std"] += constant_std
            distribution = norm(params["mean"], params["std"])

        else:
            distribution = norm(params["mean"], np.sqrt(params["variance"]))
    elif distr == "univar_lognormal":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
    elif distr == "univar_lognorm_add1_loc0":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
        if logcdf:
            probas = distribution.logsf(pred_scores_arr + 1.0)
        else:
            probas = distribution.logpdf(pred_scores_arr + 1.0)
    elif distr == "chi":
        assert "df" in params.keys(), "The number of degrees of freedom is missing, we can't define the distribution"
        df = params["df"]
        distribution = chi(df)
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        assert ("mean" in params.keys() and "std" in params.keys()), "The mean and/or standard deviation are missing, \
        we can't define the distribution"
        distribution = norm(params["mean"], params["std"])

    if probas is None:
        if logcdf:
            probas = distribution.logsf(pred_scores_arr)
        else:
            probas = distribution.logpdf(pred_scores_arr)

    return probas


def fit_univar_distr(scores_arr: np.ndarray, distr='univar_gaussian'):
    """
    :param scores_arr: 1d array of reconstruction errors
    :param distr: the name of the distribution to be fitted to anomaly scores on train data
    :return: params dict with distr name and parameters of distribution
    """
    distr_params = {'distr': distr}
    constant_std = 0.000001
    if distr == "univar_gaussian":
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std == 0.0:
            std += constant_std
        distr_params["mean"] = mean
        distr_params["std"] = std
    elif distr == "univar_lognormal":
        shape, loc, scale = lognorm.fit(scores_arr)
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "univar_lognorm_add1_loc0":
        shape, loc, scale = lognorm.fit(scores_arr + 1.0, floc=0.0)
        if shape == 0.0:
            shape += constant_std
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "chi":
        estimated_df = chi.fit(scores_arr)[0]
        df = round(estimated_df)
        distr_params["df"] = df
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        distr_params["mean"] = mean
        distr_params["std"] = std
    return distr_params


def save_torch_algo(algo: Algorithm, out_dir):
    """
    Save the trained model and the hyper parameters of the algorithm
    :param algo: the algorithm object
    :param out_dir: path to the directory where everything is to be saved
    :return: Nothing
    """
    if isinstance(algo.model, List):
        saved_model_filename = []
        for k in range(len(algo.model)):
            model_filename = os.path.join(out_dir, "trained_model_channel_%i" % k)
            saved_model_filename.append(model_filename)
            torch.save(algo.model[k], model_filename)
    else:
        saved_model_filename = os.path.join(out_dir, "trained_model")
        torch.save(algo.model, saved_model_filename)
    init_params = algo.init_params
    algo_config_filename = os.path.join(out_dir, "init_params")
    with open(algo_config_filename, "wb") as file:
        pickle.dump(init_params, file)

    additional_params_filename = os.path.join(out_dir, "additional_params")
    additional_params = algo.additional_params
    with open(additional_params_filename, "wb") as file:
        pickle.dump(additional_params, file)

    return saved_model_filename, algo_config_filename, additional_params_filename


def get_chan_num(abs_filename):
    return int(abs_filename.split("_")[-1])


def get_filenames_for_load(out_dir):
    algo_config_filename = os.path.join(out_dir, "init_params")
    saved_model_filename = [os.path.join(out_dir, filename) for filename in
                            os.listdir(out_dir) if "trained_model" in filename]
    if len(saved_model_filename) == 1:
        saved_model_filename = saved_model_filename[0]
    else:
        saved_model_filename.sort(key=get_chan_num)
    additional_params_filename = os.path.join(out_dir, "additional_params")
    return algo_config_filename, saved_model_filename, additional_params_filename


def load_torch_algo(algo_class, algo_config_filename, saved_model_filename, additional_params_filename, eval=True):
    """
    :param algo_class: Class of the Algorithm to be instantiated
    :param algo_config_filename: path to the file containing the initialization parameters of the algorithm
    :param saved_model_filename: path (or list of paths) to the trained model(s)
    :param additional_params_filename: path to the file containing the trained parameters of the algorithm eg mean, var
    :param eval: boolean to determine if model is to be put in evaluation mode
    :return: object of algo_class with a trained model
    """

    with open(os.path.join(algo_config_filename), "rb") as file:
        init_params = pickle.load(file)

    with open(additional_params_filename, "rb") as file:
        additional_params = pickle.load(file)

    # init params must contain only arguments of algo_class's constructor
    algo = algo_class(**init_params)
    device = algo.device

    if additional_params is not None:
        setattr(algo, "additional_params", additional_params)

    if isinstance(saved_model_filename, List):
        algo.model = [torch.load(path, map_location=device) for path in saved_model_filename]
        if eval:
            [model.eval() for model in algo.model]
    else:
        algo.model = torch.load(saved_model_filename, map_location=device)
        if eval:
            algo.model.eval()
    return algo

