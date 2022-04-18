import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.algorithms.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo, load_torch_algo, get_sub_seqs, \
    get_train_data_loaders,fit_with_early_stopping, predict_test_scores
from sklearn.decomposition import PCA
import warnings

"""Adapted from https://github.com/KDD-OpenSource/DeepADoTS (MIT License)"""

class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str='LSTM-ED', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_val_percentage: float=0.25,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 seed: int=None, gpu: int = None, details=True, patience: int=5, stride: int=1, out_dir=None,
                 pca_comp=None, last_t_only=True, explained_var=None, set_hid_eq_pca=False):
        """ If set_hid_eq_pca is True and one of pca_comp or explained_var is true, then hidden_size is ignored.
        Hidden size is set equal to number of pca components obtained.
        """
        if not last_t_only:
            name = "LSTM-ED_recon_all"
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)

        if set_hid_eq_pca:
            if pca_comp is not None or explained_var is not None:
                warnings.warn("set_hid_eq_pca is True and pca params provided. So hidden_size argument will be ignored. "
                              "Hidden size will be set equal to number of pca components")
                self.hidden_size = None
            else:
                set_hid_eq_pca = False
                warnings.warn("set_hid_eq_pca is True but pca params not provided. "
                              "So hidden_size argument will be used.")
                if hidden_size is None:
                    hidden_size = 5
                    warnings.warn("Hidden size was specified as None. Will use default value 5")

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.model = None
        self.torch_save = True
        # self.scaler = None
        # self.pca = None
        self.pca_comp = pca_comp
        self.explained_var = explained_var
        self.last_t_only = last_t_only
        self.set_hid_eq_pca = set_hid_eq_pca
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "hidden_size": hidden_size,
                            "n_layers": n_layers,
                            "use_bias": use_bias,
                            "dropout": dropout,
                            "sequence_length": sequence_length,
                            "stride": stride,
                            "train_val_percentage": train_val_percentage,
                            "seed": seed,
                            "gpu": gpu,
                            "details": details,
                            "patience": patience,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp,
                            "explained_var": explained_var,
                            "last_t_only": last_t_only,
                            "set_hid_eq_pca": set_hid_eq_pca
                            }

        self.additional_params = dict()
        if (pca_comp is not None) and (explained_var is not None):
            warnings.warn("Specify only one of pca_comp and explained_var.\
                PCA with pca_comp components will be implemented.")

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        pca = None

        if self.pca_comp is not None:
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
        elif self.explained_var is not None:
            pca = PCA(n_components=self.explained_var, svd_solver='full')
        if pca is not None:
            pca.fit(data_scaled)
            self.additional_params["scaler"] = scaler
            self.additional_params["pca_expl_var"] = pca.explained_variance_
            self.additional_params["pca_n_comp"] = pca.n_components_
            data = pca.transform(data_scaled)
            if self.set_hid_eq_pca:
                self.hidden_size = int(pca.n_components_) # over-ride existing hidden_size param
            print("pca explained variance {} with n_comp {}".format(pca.explained_variance_, pca.n_components_))
        self.additional_params["pca"] = pca
        self.additional_params["hidden_size"] = self.hidden_size
        print("hidden size = {}".format(self.hidden_size))
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride)
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.model = LSTMEDModule(data.shape[1], self.hidden_size, self.n_layers, self.use_bias, self.dropout,
                                  seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, last_t_only=self.last_t_only,
                                    ret_best_val_loss=True)
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params["val_reconstr_errors"] = val_reconstr_errors
        self.additional_params["best_val_loss"] = best_val_loss
    
    def fit_sequences(self, sequences):
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.model = LSTMEDModule(sequences.shape[-1], self.hidden_size, self.n_layers, self.use_bias, self.dropout,
                                  seed=self.seed, gpu=self.gpu)
        self.model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, last_t_only=self.last_t_only,
                                    ret_best_val_loss=True)
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
    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        pca = self.additional_params["pca"]
        if pca is not None:
            scaler = self.additional_params["scaler"]
            data_scaled = scaler.transform(data)
            data = pca.transform(data_scaled)
            data = data.reshape(data_scaled.shape[0], -1)

        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        error_tc, recons_tc = predict_test_scores(self.model, test_loader, latent=False, return_output=True)
        error_tc = error_tc.reshape(data_scaled.shape[0], -1)
        recons_tc = recons_tc.reshape(data_scaled.shape[0], -1)
        if pca is not None:
            recons_tc = np.dot(recons_tc, pca.components_)
            error_tc = (data_scaled - recons_tc) ** 2

        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': error_tc,
                           'recons_tc': recons_tc,
                           }
        return predictions_dic


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool=False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output


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
    algo = LSTMED(num_epochs=1, seed=seed, gpu=0, batch_size=64, hidden_size=None,
                  stride=10, train_val_percentage=0.25, explained_var=0.9, set_hid_eq_pca=True,)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])


if __name__ == "__main__":
    main()
