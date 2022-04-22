import sys, os
import math
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from src.algorithms.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo, load_torch_algo, get_sub_seqs, \
    get_train_data_loaders, get_logp_from_dist, fit_with_early_stopping, fit_scores_distribution, predict_test_scores
from src.algorithms.tcn_utils.tcn_model import TemporalBlock, TemporalBlockTranspose
from sklearn.decomposition import PCA

"""TCN adapted from https://github.com/locuslab/TCN"""

class TcnED(Algorithm, PyTorchUtils):
    def __init__(self, name: str='TcnED', num_epochs: int=10, batch_size: int=32, lr: float=1e-3, sequence_length:
                 int=55, num_channels: List=None, kernel_size: int=5, dropout: float=0.2,
                 train_val_percentage: float=0.10, seed: int=None, gpu: int=None, details=False, patience: int=2,
                 stride: int=1, out_dir=None, pca_comp=None):
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        np.random.seed(seed)
        self.torch_save = True
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.patience = patience
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.pca_comp = pca_comp
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "num_channels": num_channels,
                            "kernel_size": kernel_size,
                            "dropout": dropout,
                            "sequence_length": sequence_length,
                            "stride": stride,
                            "train_val_percentage": train_val_percentage,
                            "seed": seed,
                            "gpu": gpu,
                            "details": details,
                            "patience": patience,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp
                            }

        self.additional_params = dict()

    def __str__(self):
        return str(self.init_params)


    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            # Project input data on a limited number of principal components
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
            pca.fit(data)
            self.additional_params["pca"] = pca
            data = pca.transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride)
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.model = TcnEDModule(seq_len=self.sequence_length, num_inputs=data.shape[1], num_channels=self.num_channels,
                                 kernel_size=self.kernel_size, dropout=self.dropout, seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, ret_best_val_loss=True)
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params['val_reconstr_errors'] = val_reconstr_errors
        self.additional_params["best_val_loss"] = best_val_loss

    def fit_sequences(self, train_seqs, val_seqs):
        # X.interpolate(inplace=True)
        # X.bfill(inplace=True)
        # data = X.values
        # if self.pca_comp is not None:
        #     # Project input data on a limited number of principal components
        #     pca = PCA(n_components=self.pca_comp, svd_solver='full')
        #     pca.fit(data)
        #     self.additional_params["pca"] = pca
        #     data = pca.transform(data)
        # sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride)
        # train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
        #                                                         splits=[1 - self.train_val_percentage,
        #                                                                 self.train_val_percentage], seed=self.seed)
        train_loader = DataLoader(dataset=train_seqs, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
        train_val_loader = DataLoader(dataset=val_seqs, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)

        self.model = TcnEDModule(seq_len=self.sequence_length, num_inputs=train_seqs.shape[-1], num_channels=self.num_channels,
                                 kernel_size=self.kernel_size, dropout=self.dropout, seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, ret_best_val_loss=True)
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params['val_reconstr_errors'] = val_reconstr_errors
        self.additional_params["best_val_loss"] = best_val_loss

    def get_val_loss(self):
        try:
            val_loss = self.additional_params["best_val_loss"]
        except:
            print("could not get val_loss. Returning None")
            val_loss = None
        return val_loss        

    def get_val_err(self):
        try:
            val_err = self.additional_params["val_reconstr_errors"]
        except:
            print("could not get val_reconstr_errors. Returning None")
            val_err = None
        return val_err

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            data = self.additional_params["pca"].transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array  = predict_test_scores(self.model, test_loader, latent=False, return_output=True)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic

    @torch.no_grad()
    def predict_sequences(self, sequences) -> np.array:
        # X.interpolate(inplace=True)
        # X.bfill(inplace=True)
        # data = X.values
        # if self.pca_comp is not None:
        #     data = self.additional_params["pca"].transform(data)
        # sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array  = predict_test_scores(self.model, test_loader, latent=False, return_output=True)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic


class TcnEDModule(nn.Module, PyTorchUtils):
    def __init__(self, seq_len:int, num_inputs:int, num_channels:List, seed:int, gpu:int, kernel_size=2, dropout=0.2):
        super(TcnEDModule, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.encoder_layers = []
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.num_inputs = num_inputs
        num_levels = len(num_channels)

        # encoder
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.encoder_layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding_size, dropout=dropout)]

        # decoder
        decoder_channels = list(reversed(num_channels))
        self.decoder_layers = []
        for i in range(num_levels):
            # no dilation in decoder
            in_channels = decoder_channels[i]
            out_channels = num_inputs if i==(num_levels-1) else decoder_channels[i+1]
            dilation_size = 2 ** (num_levels-1-i)
            padding_size = (kernel_size-1) * dilation_size
            self.decoder_layers += [TemporalBlockTranspose(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding_size, dropout=dropout)]

        # to register parameters in list of layers, each layer must be an object
        self.enc_layer_names = ["enc_" + str(num) for num in range(len(self.encoder_layers))]
        self.dec_layer_names = ["dec_" + str(num) for num in range(len(self.decoder_layers))]
        for name, layer in zip(self.enc_layer_names, self.encoder_layers):
            setattr(self, name, layer)
        for name, layer in zip(self.dec_layer_names, self.decoder_layers):
            setattr(self, name, layer)

    def forward(self, x, return_latent=False):
        out = x.permute(0, 2, 1)
        enc = nn.Sequential(*self.encoder_layers)(out)
        dec = nn.Sequential(*self.decoder_layers)(enc)     
        if return_latent:
            return dec.permute(0, 2, 1), enc
        else:
            return dec.permute(0, 2, 1)


def main():
    from src.datasets.skab import Skab
    seed = 0
    print("Running main")
    ds = Skab(seed=seed)
    x_train, y_train, x_test, y_test = ds.data()
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    algo = TcnED(sequence_length=30, num_epochs=10, num_channels=[5]*3, lr=1e-3, gpu=0, batch_size=16, pca_comp=5, train_val_percentage=0.25, stride=10)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])


if __name__ == "__main__":

    main()
