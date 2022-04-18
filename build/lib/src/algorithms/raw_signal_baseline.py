from src.algorithms.algorithm_utils import Algorithm
import numpy as np
import pandas as pd


class RawSignalBaseline(Algorithm):
    def __init__(self, name: str='RawSignalBaseline', seed: int=None, details=True, out_dir=None,
                 train_starts=np.array([])):
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        self.model = None
        self.init_params = {"name": name,
                            "seed": seed,
                            "details": details,
                            "out_dir": out_dir,
                            "train_starts": train_starts}
        self.torch_save = False

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': data,
                           'recons_tc': np.zeros(data.shape)
                           }
        return predictions_dic
