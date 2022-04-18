from src.algorithms.algorithm_utils import Algorithm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PcaRecons(Algorithm):
    def __init__(self, name: str='PcaRecons', explained_var=0.9, seed: int=None, details=True, out_dir=None,
                 train_starts=np.array([])):
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        self.explained_var = explained_var
        self.init_params = {"name": name,
                            "explained_var": explained_var,
                            "seed": seed,
                            "details": details,
                            "out_dir": out_dir,
                            "train_starts": train_starts
                            }
        self.scaler = None
        self.model = None
        self.additional_params = dict()
        self.torch_save = False

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        self.scaler = StandardScaler()

        data_scaled = self.scaler.fit_transform(data)

        self.model = PCA(n_components=self.explained_var)
        print("Fitting PCA")
        self.model.fit(data_scaled)
        print("Done fitting PCA. ncomponents for explained variance {} = {}".format(self.explained_var,
                                                                                    self.model.n_components_))
        self.additional_params["scaler"] = self.scaler
        self.additional_params["model"] = self.model

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        data_scaled = self.scaler.transform(data)
        recons_tc = np.dot(self.model.transform(data_scaled), self.model.components_)
        error_tc = (data_scaled - recons_tc) ** 2
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': error_tc,
                           'recons_tc': recons_tc
                           }
        return predictions_dic
