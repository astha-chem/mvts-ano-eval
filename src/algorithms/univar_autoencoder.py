import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from src.algorithms.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo, load_torch_algo, get_sub_seqs, \
    get_train_data_loaders, fit_with_early_stopping, predict_test_scores
from src.algorithms.autoencoder import AutoEncoderModule


class UnivarAutoEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str='UnivarAutoEncoder_v1', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_val_percentage: float=0.25,
                 seed: int=None, gpu: int=None, details=True, patience: int=5, stride: int=1, out_dir=None,
                 n_processes=1, train_starts=np.array([]), last_t_only=True):
        if not last_t_only:
            name = "UnivarAutoEncoder_recon_all"
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        self.torch_save = True
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.n_processes = n_processes
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
                            "train_starts": train_starts,
                            "last_t_only": last_t_only
                            }

        self.additional_params = dict()

    @staticmethod
    def fit_one_channel(channel_data, channel_num, out_dir, params):
        sequences = get_sub_seqs(channel_data, seq_len=params['sequence_length'], stride=params['stride'],
                                 start_discont=params["train_starts"])
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=params['batch_size'],
                                                                splits=[1 - params['train_val_percentage'],
                                                                        params['train_val_percentage']],
                                                                seed=params['seed'])
        model = AutoEncoderModule(1, params['sequence_length'], params['hidden_size'], seed=params['seed'],
                                                    gpu=params['gpu'])

        model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, model, patience=params['patience'],
                                    num_epochs=params['num_epochs'], lr=params['lr'], last_t_only=params["last_t_only"],
                                    ret_best_val_loss=True)
        model_filename = os.path.join(out_dir, "trained_model_channel_%i" % channel_num)
        torch.save(model, model_filename)
        return model_filename, channel_num, train_loss, val_loss, val_reconstr_errors, best_val_loss

    @staticmethod
    def fit_one_channel_sequences(train_seqs, val_seqs, channel_num, out_dir, params):
        train_loader = DataLoader(dataset=train_seqs, batch_size=params['batch_size'], drop_last=False, pin_memory=True, shuffle=False)
        train_val_loader = DataLoader(dataset=val_seqs, batch_size=params['batch_size'], drop_last=False, pin_memory=True, shuffle=False)

        model = AutoEncoderModule(1, params['sequence_length'], params['hidden_size'], seed=params['seed'],
                                                    gpu=params['gpu'])

        model, train_loss, val_loss, val_reconstr_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, model, patience=params['patience'],
                                    num_epochs=params['num_epochs'], lr=params['lr'], last_t_only=params["last_t_only"],
                                    ret_best_val_loss=True)

        
        model_filename = os.path.join(out_dir, "trained_model_channel_%i" % channel_num)
        torch.save(model, model_filename)
        return model_filename, channel_num, train_loss, val_loss, val_reconstr_errors, best_val_loss

    def predict_one_channel(self, channel_data, channel_num, starts_discont=np.array([])):
        sequences = get_sub_seqs(channel_data, seq_len=self.sequence_length, stride=1, start_discont=starts_discont)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model[channel_num], test_loader,
                                                             return_output=True)
        return (reconstr_errors, outputs_array)

    def predict_one_channel_sequences(self, sequences, channel_num, starts_discont=np.array([])):
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model[channel_num], test_loader,
                                                             return_output=True)
        return (reconstr_errors, outputs_array)

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        self.additional_params["train_channels"] = data.shape[1]
        self.model = [0] * self.additional_params["train_channels"]
        self.additional_params["train_loss_per_epoch"] = []
        self.additional_params["val_loss_per_epoch"] = []
        self.additional_params["val_reconstr_errors"] = []
        self.additional_params["best_val_loss"] = []
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=self.n_processes)
        results = []
        for channel_num in range(self.additional_params["train_channels"]):
            print("Training univariate model on channel number %i" % channel_num)
            args = (data[:, channel_num].reshape(-1, 1), channel_num, self.out_dir, self.init_params)
            results.append(pool.apply_async(self.fit_one_channel, args=args))
        return_values = [result.get() for result in results]
        pool.close()
        pool.terminate()
        pool.join()
        # print("results: {}".format(roots[:-1]))
        for return_value in return_values:
            model_filename, channel_num, train_loss, val_loss, val_reconstr_scores, best_val_loss = return_value
            self.model[channel_num] = torch.load(model_filename)
            self.additional_params["train_loss_per_epoch"].append(train_loss)
            self.additional_params["val_loss_per_epoch"].append(val_loss)
            self.additional_params["val_reconstr_errors"].append(val_reconstr_scores)
            self.additional_params["best_val_loss"].append(best_val_loss)

    def fit_sequences(self, train_seqs, val_seqs):
    
        self.additional_params["train_channels"] = train_seqs.shape[-1]
        self.model = [0] * self.additional_params["train_channels"]
        self.additional_params["train_loss_per_epoch"] = []
        self.additional_params["val_loss_per_epoch"] = []
        self.additional_params["val_reconstr_errors"] = []
        self.additional_params["best_val_loss"] = []
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=self.n_processes)
        results = []
        for channel_num in range(self.additional_params["train_channels"]):
            print("Training univariate model on channel number %i" % channel_num)
            args = (train_seqs[:,:,channel_num], val_seqs[:,:,channel_num], channel_num, self.out_dir, self.init_params)
            results.append(pool.apply_async(self.fit_one_channel_sequences, args=args))
        return_values = [result.get() for result in results]
        pool.close()
        pool.terminate()
        pool.join()
        # print("results: {}".format(roots[:-1]))
        best_val_losses = 0.0
        for return_value in return_values:
            model_filename, channel_num, train_loss, val_loss, val_reconstr_scores, best_val_loss = return_value
            self.model[channel_num] = torch.load(model_filename)
            self.additional_params["train_loss_per_epoch"].append(train_loss)
            self.additional_params["val_loss_per_epoch"].append(val_loss)
            self.additional_params["val_reconstr_errors"].append(val_reconstr_scores)
            self.additional_params["best_val_loss"].append(best_val_loss)
            best_val_losses += best_val_loss

        return np.sum(best_val_losses)

    # def fit_sequences(self, train_seqs, val_seqs):
    #     # X.interpolate(inplace=True)
    #     # X.bfill(inplace=True)
    #     # data = X.values
    #     self.additional_params["train_channels"] = train_seqs.shape[-1]
    #     if self.model is None:
    #         self.model = [None] * self.additional_params["train_channels"]
    #         self.best_val_loss = [None] * self.additional_params["train_channels"]                                              

    #     self.additional_params["train_loss_per_epoch"] = []
    #     self.additional_params["val_loss_per_epoch"] = []
    #     self.additional_params["val_reconstr_errors"] = []
    #     self.additional_params["best_val_loss"] = []
    #     ctx = mp.get_context('spawn')
    #     # pool = ctx.Pool(processes=self.n_processes)
    #     results = []
    #     for channel_num in range(self.additional_params["train_channels"]):
    #         print("Training univariate model on channel number %i" % channel_num)
    #         args = (self.model[channel_num], self.best_val_loss[channel_num], train_seqs[:,:,channel_num], val_seqs[:,:,channel_num], channel_num, self.out_dir, self.init_params)
    #         results.append(self.fit_one_channel_sequences(*args))
    #     # return_values = [result.get() for result in results]
    #     # pool.close()
    #     # pool.terminate()
    #     # pool.join()
    #     # print("results: {}".format(roots[:-1]))
    #     any_model_changed = False
    #     for return_value in results:
    #         model_filename, channel_num, train_loss, val_loss, val_reconstr_scores, best_val_loss, model_changed = return_value
    #         self.model[channel_num] = torch.load(model_filename)
    #         self.additional_params["train_loss_per_epoch"].append(train_loss)
    #         self.additional_params["val_loss_per_epoch"].append(val_loss)
    #         self.additional_params["val_reconstr_errors"].append(val_reconstr_scores)
    #         self.additional_params["best_val_loss"].append(best_val_loss)
    #         self.best_val_loss[channel_num] = best_val_loss
    #         if model_changed:
    #             any_model_changed = True
    #     return np.sum(self.best_val_loss), any_model_changed

    def get_val_loss(self):
        try:
            val_loss = sum(self.additional_params["best_val_loss"])
        except:
            print("could not get val_err. Returning None")
            val_loss = None
        return val_loss

    def get_val_err(self):
        try:
            val_err = np.sum(np.stack(self.additional_params["val_reconstr_errors"], axis=-1), axis=1)
        except:
            print("could not get val_reconstr_errors_tc. Returning None")
            val_err = None
        return val_err

    def save_model_get_filename(self):
        saved_model_filename, algo_config_filename, additional_params_filename = save_torch_algo(self,
                                                                                                 out_dir=self.out_dir)
        return saved_model_filename, algo_config_filename, additional_params_filename

    @torch.no_grad()
    def predict(self, X: pd.DataFrame, starts=np.array([])) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        intermediate_scores = []
        outputs_array = []
        for channel_num in range(self.additional_params["train_channels"]):
            chan_scores, chan_outputs = self.predict_one_channel(data[:, channel_num].reshape(-1, 1), channel_num,
                                                                 starts_discont=starts)
            chan_scores = chan_scores.reshape((-1))
            chan_outputs = chan_outputs.reshape((-1))
            intermediate_scores.append(chan_scores)
            outputs_array.append(chan_outputs)
        intermediate_scores = np.array(intermediate_scores).T
        outputs_array = np.array(outputs_array).T
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': intermediate_scores,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic
    
    @torch.no_grad()
    def predict_sequences(self, sequences):
        # X.interpolate(inplace=True)
        # X.bfill(inplace=True)
        # data = X.values
        intermediate_scores = []
        outputs_array = []
        for channel_num in range(self.additional_params["train_channels"]):
            chan_scores, chan_outputs = self.predict_one_channel_sequences(sequences[:,:, channel_num], channel_num,
                                                                 starts_discont=np.array([]))
            chan_scores = chan_scores.reshape((-1))
            chan_outputs = chan_outputs.reshape((-1))
            intermediate_scores.append(chan_scores)
            outputs_array.append(chan_outputs)
        intermediate_scores = np.array(intermediate_scores).T
        outputs_array = np.array(outputs_array).T
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': intermediate_scores,
                           'recons_tc': outputs_array,
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
    algo = UnivarAutoEncoder(sequence_length=30, num_epochs=1, hidden_size=15, lr=1e-3, gpu=0, batch_size=512)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["error_tc"].shape)
    print(results["error_tc"][:10])

if __name__ == "__main__":

    main()
