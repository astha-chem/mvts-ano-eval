import sys, os
import numpy as np
import pandas as pd
from thundersvm import OneClassSVM
from sklearn.decomposition import PCA
from src.algorithms.algorithm_utils import Algorithm, get_sub_seqs, get_train_data_loaders
import warnings

# one class svm
# implemented on each channel if univar == True, else concatenate all channels
# hyperparameters as in sklearn OneClassSVM
# difference from SVM1c: gpu version, default gamma = 'auto', score is distance from hyperplane since score_samples() is unavailable
# note that batch_size is not used to fit, but to ensure same training/validation sets are loaded
class SVM1c_gpu(Algorithm):
    def __init__(self, name: str='SVM1c_gpu', kernel='rbf', degree=3, gamma='auto',
        coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, 
        sequence_length: int=30, seed: int=None, details=True, stride: int=1, out_dir=None, pca_comp=None, explained_var=0.9,
        train_starts=np.array([]), univar=False, train_val_percentage=0.25, batch_size: int=100):
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        
        self.model = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.sequence_length = sequence_length
        self.seed = seed
        self.stride = stride
        self.pca_comp = pca_comp
        self.explained_var = explained_var
        self.train_starts = train_starts
        self.univar = univar
        self.train_val_percentage = train_val_percentage
        self.batch_size = batch_size
        self.init_params = {"name": name,
                            "kernel": kernel,
                            "degree": degree,
                            "gamma": gamma,
                            "coef0": coef0,
                            "tol": tol,
                            "nu": nu,
                            "shrinking": shrinking,
                            "cache_size": cache_size,
                            "verbose": verbose,
                            "max_iter": max_iter,
                            "sequence_length": sequence_length,
                            "seed": seed,
                            "details": details,
                            "stride": stride,
                            "out_dir": out_dir,
                            "pca_comp": pca_comp,
                            "explained_var": explained_var,
                            "train_starts": train_starts,
                            "univar":univar,
                            "train_val_percentage":train_val_percentage,
                            "batch_size":batch_size
                            }
        self.additional_params = dict()
        if (pca_comp is not None) and (explained_var is not None):
            warnings.warn("Specify only one of pca_comp and explained_var.\
                PCA with pca_comp components will be implemented.")

    def convertpred(self, s):
        '''
        svm's predict function returns -1 or outlier and 1 for inlier
        transforming to 1 for outlier and 0 for inlier
        '''
        return (-s+1.0)/2


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
        elif self.explained_var is not None:
            # Project input data on a limited number of principal components based on percentage of variance explained
            pca = PCA(n_components=self.explained_var, svd_solver='full')
            pca.fit(data)
            self.additional_params["pca"] = pca
            self.additional_params["pca_expl_var"] = pca.explained_variance_
            self.additional_params["pca_n_comp"] = len(pca.explained_variance_)
            data = pca.transform(data)

        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride,
                                 start_discont=self.train_starts)
        train_sequences, val_sequences = get_train_data_loaders(sequences, batch_size=self.batch_size,
            splits=[1 - self.train_val_percentage, self.train_val_percentage], seed=self.seed,
            usetorch = False)
        if self.univar:
            # fit model on each channel
            model = [None]*data.shape[1]
            for n in range(data.shape[1]):
                train_sequences_n = np.asarray([s[:,n] for s in train_sequences])
                model[n] = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                    coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking, 
                    cache_size=self.cache_size, verbose=self.verbose, max_iter=self.max_iter)
                model[n].fit(train_sequences_n)
        else:
            # concatenate channels
            train_sequences = train_sequences.reshape(train_sequences.shape[0],-1)
            model = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                    coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking, 
                    cache_size=self.cache_size, verbose=self.verbose, max_iter=self.max_iter)
            model.fit(train_sequences)
        self.model = model
        # validation scores
        if self.univar:
            # predict using model on each channel
            score_tc = [None]*data.shape[1]
            for n in range(data.shape[1]):
                val_sequences_n = np.asarray([s[:,n] for s in val_sequences])
                # binary score
                score_tc[n] = self.convertpred(self.model[n].predict(val_sequences_n).reshape(-1))
            self.additional_params['val_score_t'] = np.asarray(score_tc).transpose().mean(axis = 1)
            if self.pca_comp is None:
                self.additional_params['val_score_tc'] = np.asarray(score_tc).transpose()
        else:
            # concatenate channels
            val_sequences = val_sequences.reshape(val_sequences.shape[0],-1)
            # binary score
            score_t = self.convertpred(self.model.predict(val_sequences).reshape(-1))
            self.additional_params['val_score_t'] = score_t


    def predict(self, X: pd.DataFrame, starts=np.array([])) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if (self.pca_comp is not None) or (self.explained_var is not None):
            data = self.additional_params["pca"].transform(data)

        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1, start_discont=starts)
        padding = np.zeros(self.sequence_length-1)
        
        if self.univar:
            # predict using model on each channel
            score_tc = [None]*data.shape[1]
            for n in range(data.shape[1]):
                sequences_n = np.asarray([s[:,n] for s in sequences])
                # binary score
                score_tc[n] = np.concatenate([padding, self.convertpred(self.model[n].predict(sequences_n).reshape(-1))])
            if self.pca_comp is not None:
                final_score_tc = None
            else:
                final_score_tc = np.asarray(score_tc).transpose()
            predictions_dic = {'score_t': np.asarray(score_tc).transpose().mean(axis = 1),
                   'score_tc': final_score_tc,
                   'error_t': None,
                   'error_tc': None,
                   'recons_tc': None,
                   }
        else:
            # concatenate channels
            sequences = sequences.reshape(sequences.shape[0],-1)
            # binary score
            score_t = self.convertpred(self.model.predict(sequences).reshape(-1))
            score_t = np.concatenate([padding, score_t])
            predictions_dic = {'score_t': score_t,
                               'score_tc': None,
                               'error_t': None,
                               'error_tc': None,
                               'recons_tc': None,
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
    # algo = SVM1c_gpu(sequence_length=30, pca_comp=5, stride=10, univar=False, batch_size=16, nu=0.1)
    algo = SVM1c_gpu(sequence_length=30, explained_var=0.9, stride=10, univar=False, batch_size=16, nu=0.1)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["score_t"].shape)
    print(results["score_t"][:10])


if __name__ == "__main__":

    main()