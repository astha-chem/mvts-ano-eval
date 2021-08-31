import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.algorithms.mscred_utils.mscred_model import MSCREDModule
from sklearn.decomposition import PCA
from src.algorithms.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo, load_torch_algo, get_sub_seqs, \
    get_train_data_loaders, get_logp_from_dist, fit_scores_distribution
from src.algorithms.mscred_utils.utils import fit_with_early_stopping, predict_test_scores  
import warnings

"""MSCRED adapted from https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED, https://github.com/SKvtun/MSCRED-Pytorch (MIT License)"""

# MSCRED
# hyperparameters as in MSCRED
class MSCRED(Algorithm, PyTorchUtils):
    def __init__(self, name: str='MSCRED', step_max: int=5, 
        num_epochs: int=200, batch_size: int=100, lr: float=1e-3, verbose=False,
        sequence_length: int=30, seed: int=None, details=True, stride: int=1, out_dir=None, pca_comp=None, explained_var=None,
        train_starts=np.array([]), gpu: int=None, train_val_percentage=0.25, patience=10, model_id=0):
        """
        Args to initialize MSCRED:
        step_max = 5 # maximum step of ConvLSTM 
        win_size = sequence_length/step_max
        """
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        np.random.seed(seed)
        self.torch_save = False
        self.model = None
        self.step_max = step_max
        self.win_size = [int((sequence_length/step_max)*i/3) for i in range(1,4)] # 3 windows
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.sequence_length = sequence_length
        self.seed = seed
        self.stride = stride
        self.pca_comp = pca_comp
        self.explained_var = explained_var
        self.train_starts = train_starts
        self.gpu = gpu
        self.train_val_percentage = train_val_percentage
        self.patience = patience
        self.model_id = model_id
        self.init_params = {"name": name,
                            "step_max": step_max,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "verbose": verbose,
                            "sequence_length": sequence_length,
                            "seed": seed,
                            "details": details,
                            "stride": stride,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp,
                            "explained_var": explained_var,
                            "train_starts": train_starts,
                            "gpu": gpu,
                            "train_val_percentage": train_val_percentage,
                            "patience": patience
                            }
        self.additional_params = dict()


    def fit(self, X: pd.DataFrame):

        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        pca = None
        if self.pca_comp is not None:
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
        elif self.explained_var is not None:
            pca = PCA(n_components=self.explained_var, svd_solver='full')        
        self.additional_params["pca"] = pca
        if pca is not None:
            # Project input data on a limited number of principal components
            pca.fit(data)
            self.additional_params["pca_expl_var"] = pca.explained_variance_
            self.additional_params["pca_n_comp"] = pca.n_components_
            data = pca.transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride,
                                 start_discont=self.train_starts) # n_samp x timesteps x n_dim
        # create signature matrices
        n_dim = sequences.shape[2]
        matrices = np.zeros((sequences.shape[0], 3, self.step_max,
            n_dim, n_dim))
        for i in range(sequences.shape[0]):
            raw_data_i = sequences[i]
            for k,w in enumerate(self.win_size):
                pad = self.sequence_length - self.step_max*w
                for j in range(self.step_max):
                    raw_data_ij = raw_data_i[(pad+j*w):(pad+(j+1)*w)]
                    matrices[i,k,j] = np.dot(raw_data_ij.T, raw_data_ij) / w

        train_loader, train_val_loader = get_train_data_loaders(matrices, batch_size=self.batch_size,
            splits=[1 - self.train_val_percentage, self.train_val_percentage], seed=self.seed)

        self.model = MSCREDModule(num_timesteps=self.step_max, attention=True, seed=self.seed, gpu=self.gpu)
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
        pca = self.additional_params["pca"]        
        if pca is not None:
            data = pca.transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1)
        n_dim = sequences.shape[2]
        matrices = np.zeros((sequences.shape[0], 3, self.step_max,
            n_dim, n_dim))
        for i in range(sequences.shape[0]):
            raw_data_i = sequences[i]
            for k,w in enumerate(self.win_size):
                pad = self.sequence_length - self.step_max*w
                for j in range(self.step_max):
                    raw_data_ij = raw_data_i[(pad+j*w):(pad+(j+1)*w)]
                    matrices[i,k,j] = np.dot(raw_data_ij.T, raw_data_ij) / w
        test_loader = DataLoader(dataset=matrices, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors  = predict_test_scores(self.model, test_loader)
        padding = np.zeros((self.sequence_length-1, data.shape[1]))
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': np.vstack((padding, reconstr_errors)),
                           'recons_tc': None
                           }
        return predictions_dic


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
    algo = MSCRED(sequence_length=30, num_epochs=2, lr=1e-3, gpu=0, batch_size=16, explained_var=0.9, stride=10, patience=2)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])


if __name__ == "__main__":

    main()
