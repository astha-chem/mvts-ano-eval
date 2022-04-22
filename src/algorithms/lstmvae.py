import sys, os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from src.algorithms.lstm_vae.lstm_vae_model import LSTM_Var_Autoencoder
from sklearn.decomposition import PCA
from src.algorithms.algorithm_utils import Algorithm, get_sub_seqs, TensorflowUtils, get_train_data_loaders
import warnings
import ipdb

"""Variational LSTM Autoencoder adapted from https://github.com/Danyleb/Variational-Lstm-Autoencoder"""

# variational LSTM autoencoder
# hyperparameters as in LSTM_Var_Autoencoder
class VAE_LSTM(Algorithm, TensorflowUtils):
    def __init__(self, name: str='VAE_LSTM', intermediate_dim=None, z_dim=None, n_dim=None, kulback_coef=0.1, stateful=False,
        num_epochs: int=200, batch_size: int=100, lr: float=1e-3, REG_LAMBDA=0, grad_clip_norm=10, optimizer_params=None, verbose=False,
        sequence_length: int=30, seed: int=None, details=True, stride: int=1, out_dir=None, pca_comp=None,
        train_starts=np.array([]), gpu: int=None, train_val_percentage=0.25, patience=10, model_id=0):
        """
        Args to initialize LSTM_Var_Autoencoder:
        intermediate_dim : LSTM cells dimension. # num_units in LSTMCell
        z_dim : dimension of latent space. # latent space sums across timesteps from LSTM
        n_dim : dimension of input data.
        statefull : if true, keep cell state through batches.
        """
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        TensorflowUtils.__init__(self, seed, gpu)
        self.torch_save = False
        self.model = None
        self.intermediate_dim = intermediate_dim
        self.z_dim = z_dim
        self.n_dim = n_dim
        self.kulback_coef = kulback_coef
        self.stateful = stateful
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.REG_LAMBDA = REG_LAMBDA
        self.grad_clip_norm = grad_clip_norm
        self.optimizer_params = optimizer_params
        self.verbose = verbose
        self.sequence_length = sequence_length
        self.seed = seed
        self.stride = stride
        self.pca_comp = pca_comp
        self.train_starts = train_starts
        self.gpu = gpu
        self.train_val_percentage = train_val_percentage
        self.patience = patience
        self.model_id = model_id
        self.init_params = {"name": name,
                            "intermediate_dim": intermediate_dim,
                            "z_dim": z_dim,
                            "n_dim": n_dim,
                            "kulback_coef": kulback_coef,
                            "stateful": stateful,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "REG_LAMBDA": REG_LAMBDA,
                            "grad_clip_norm": grad_clip_norm,
                            "optimizer_params": optimizer_params,
                            "verbose": verbose,
                            "sequence_length": sequence_length,
                            "seed": seed,
                            "details": details,
                            "stride": stride,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp,
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
        if self.pca_comp is not None:
            # Project input data on a limited number of principal components
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
            pca.fit(data)
            self.additional_params["pca"] = pca
            self.additional_params["pca_expl_var"] = pca.explained_variance_
            data = pca.transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride,
                                 start_discont=self.train_starts) # n_samp x timesteps x n_dim
        train_sequences, val_sequences = get_train_data_loaders(sequences, batch_size=self.batch_size,
            splits=[1 - self.train_val_percentage, self.train_val_percentage], seed=self.seed,
            usetorch = False)
        # fitting with early stopping
        with self.device:
            model = LSTM_Var_Autoencoder(intermediate_dim=self.intermediate_dim, z_dim=self.z_dim, n_dim=self.n_dim,
                                         stateful=self.stateful, model_id=self.model_id)
            train_loss, val_loss, best_val_loss = model.fit(train_sequences, val_sequences, learning_rate=self.lr,
                batch_size=self.batch_size, num_epochs=self.num_epochs, REG_LAMBDA=self.REG_LAMBDA,
                grad_clip_norm=self.grad_clip_norm, optimizer_params=self.optimizer_params, verbose=self.verbose,
                                                            patience=self.patience)
        self.model = model
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params["best_val_loss"] = best_val_loss
        # validation reconstruction error
        recons_error = []
        with self.device:
            num_batch = int(np.ceil(len(val_sequences)/self.batch_size))
            for i in range(num_batch):
                s = val_sequences[i*self.batch_size:(i+1)*self.batch_size]
                _, recons_error_s = self.model.reconstruct(s, get_error = True) # returns squared error
                recons_error.append(recons_error_s)
        recons_error = np.vstack(recons_error)
        # follows reconstrunction error computation in predict_test_scores in src.algorithms.algorithm_utils
        # convert to L1 loss
        # also only use recontruction and error for last point at each sequence
        recons_error = np.sqrt(recons_error) # n_samp x timesteps x n_dim
        recons_error_laststep = np.squeeze(recons_error[:,-1,:])
        self.additional_params['val_reconstr_errors'] = recons_error_laststep

    def fit_sequences(self, train_seqs, val_seqs):

        with self.device:
            if self.model is None:
                model = LSTM_Var_Autoencoder(intermediate_dim=self.intermediate_dim, z_dim=self.z_dim, n_dim=self.n_dim,
                                         stateful=self.stateful, model_id=self.model_id)
                self.last_best_val_loss = None
            else:
                model = self.model
            train_loss, val_loss, best_val_loss = model.fit(train_seqs, val_seqs, learning_rate=self.lr,
                batch_size=self.batch_size, num_epochs=self.num_epochs, REG_LAMBDA=self.REG_LAMBDA,
                grad_clip_norm=self.grad_clip_norm, optimizer_params=self.optimizer_params, verbose=self.verbose,
                                                            patience=self.patience)
        if self.last_best_val_loss is None or self.last_best_val_loss > best_val_loss:
            self.model = model
            self.last_best_val_loss = best_val_loss

        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params["best_val_loss"] = best_val_loss
        # validation reconstruction error
        recons_error = []
        with self.device:
            num_batch = int(np.ceil(len(val_seqs)/self.batch_size))
            for i in range(num_batch):
                s = val_seqs[i*self.batch_size:(i+1)*self.batch_size]
                _, recons_error_s = self.model.reconstruct(s, get_error = True) # returns squared error
                recons_error.append(recons_error_s)
        recons_error = np.vstack(recons_error)
        # follows reconstrunction error computation in predict_test_scores in src.algorithms.algorithm_utils
        # convert to L1 loss
        # also only use recontruction and error for last point at each sequence
        recons_error = np.sqrt(recons_error) # n_samp x timesteps x n_dim
        recons_error_laststep = np.squeeze(recons_error[:,-1,:])
        self.additional_params['val_reconstr_errors'] = recons_error_laststep
        return self.last_best_val_loss

    def predict_sequences(self, sequences):
        padding = np.zeros((self.sequence_length-1, sequences.shape[-1]))
        reconstructed = []
        recons_error = []
        with self.device:
            num_batch = int(np.ceil(len(sequences)/self.batch_size))
            for i in range(num_batch):
                s = sequences[i*self.batch_size:(i+1)*self.batch_size]
                reconstructed_s, recons_error_s = self.model.reconstruct(s, get_error = True) # returns squared error
                reconstructed.append(reconstructed_s)
        reconstructed = np.vstack(reconstructed)
        # only use recontruction and error for last point at each sequence
        reconstructed_laststep = np.squeeze(reconstructed[:,-1,:])
        # fill NaN with previous value for reconstruction
        reconstructed_laststep_df = pd.DataFrame(reconstructed_laststep)
        reconstructed_laststep_df.fillna(method='ffill', inplace=True)
        reconstructed_laststep = np.asarray(reconstructed_laststep_df)
        # recons_error_laststep = np.abs(data[-reconstructed_laststep.shape[0]:] - reconstructed_laststep)
        seqs = sequences[:,-1,:]
        recons_error_laststep = np.abs(seqs[-reconstructed_laststep.shape[0]:] - reconstructed_laststep)

        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': np.vstack((padding, recons_error_laststep)),
                           'recons_tc': np.vstack((padding, reconstructed_laststep))
                           }
        return predictions_dic
    
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

    def predict(self, X: pd.DataFrame, starts=np.array([])) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            data = self.additional_params["pca"].transform(data)

        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1, start_discont=starts)
        padding = np.zeros((self.sequence_length-1, data.shape[1]))
        reconstructed = []
        recons_error = []
        with self.device:
            num_batch = int(np.ceil(len(sequences)/self.batch_size))
            for i in range(num_batch):
                s = sequences[i*self.batch_size:(i+1)*self.batch_size]
                reconstructed_s, recons_error_s = self.model.reconstruct(s, get_error = True) # returns squared error
                reconstructed.append(reconstructed_s)
        reconstructed = np.vstack(reconstructed)
        # only use recontruction and error for last point at each sequence
        reconstructed_laststep = np.squeeze(reconstructed[:,-1,:])
        # fill NaN with previous value for reconstruction
        reconstructed_laststep_df = pd.DataFrame(reconstructed_laststep)
        reconstructed_laststep_df.fillna(method='ffill', inplace=True)
        reconstructed_laststep = np.asarray(reconstructed_laststep_df)
        recons_error_laststep = np.abs(data[-reconstructed_laststep.shape[0]:] - reconstructed_laststep)

        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': np.vstack((padding, recons_error_laststep)),
                           'recons_tc': np.vstack((padding, reconstructed_laststep))
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
    algo = VAE_LSTM(sequence_length=30, intermediate_dim=5, z_dim=2, n_dim=x_train.shape[1], num_epochs=100, lr=1e-3, batch_size=16, pca_comp=None, stride=10, patience=2)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])


if __name__ == "__main__":

    main()
