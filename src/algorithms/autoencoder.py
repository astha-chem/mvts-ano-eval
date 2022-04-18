import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.algorithms.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo, load_torch_algo, get_sub_seqs, \
    get_train_data_loaders, fit_with_early_stopping, predict_test_scores


class AutoEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str='AutoEncoder', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_val_percentage: float=0.25,
                 seed: int=None, gpu: int=None, details=True, patience: int=2, stride: int=1, out_dir=None, pca_comp=None,
                 train_starts=np.array([]), last_t_only=True):
        if not last_t_only:
            name = "AutoEncoder_recon_all"
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.torch_save = True
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.pca_comp = pca_comp
        self.train_starts = train_starts
        self.last_t_only = last_t_only
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "hidden_size": hidden_size,
                            "sequence_length": sequence_length,
                            "stride": stride,
                            "train_val_percentage": train_val_percentage,
                            "seed": seed,
                            "gpu": gpu,
                            "details": details,
                            "patience": patience,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp,
                            "train_starts": train_starts,
                            "last_t_only": last_t_only,
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
                                 start_discont=self.train_starts)
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.model = AutoEncoderModule(data.shape[1], self.sequence_length, self.hidden_size, seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, verbose=True, last_t_only=self.last_t_only,
                                    ret_best_val_loss=True)
        self.additional_params["val_reconstr_errors"] = val_reconstr_errors
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params["best_val_loss"] = best_val_loss

    def fit_sequences(self, sequences):
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        if self.model is None:                                                                        
            self.model = AutoEncoderModule(sequences.shape[-1], self.sequence_length, self.hidden_size, seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, verbose=True, last_t_only=self.last_t_only,
                                    ret_best_val_loss=True)
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
            print("could not get val_reconstr_errors_tc. Returning None")
            val_err = None
        return val_err

    @torch.no_grad()
    def predict(self, X: pd.DataFrame, starts=np.array([])) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            data = self.additional_params["pca"].transform(data)

        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1, start_discont=starts)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model, test_loader, latent=False,
                                                             return_output=True)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic
    
    @torch.no_grad()
    def predict_sequences(self, sequences):
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model, test_loader, latent=False,
                                                             return_output=True)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic


class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int, hidden_size: int, seed: int, gpu: int):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence


def main():
    from src.datasets.skab import Skab
    import os, time
    seed = 0
    print("Running main")
    ds = Skab(seed=seed)
    x_train, y_train, x_test, y_test = ds.data()
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]    
    algo = AutoEncoder(sequence_length=30, num_epochs=1, hidden_size=2, lr=1e-3, gpu=0, batch_size=16, pca_comp=5,
                       stride=10, train_val_percentage=0.25)
    timestamp = time.strftime('%Y-%m-%d-%H%M%S')
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])


if __name__ == "__main__":
    main()
