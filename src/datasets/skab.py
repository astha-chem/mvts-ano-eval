import math
import pandas as pd
import numpy as np
import os
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
"""
@author: Astha Garg 01/21
"""


class Skab(Dataset):

    def __init__(self, seed: int, entity=None, verbose=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        name = "skab"
        super().__init__(name=name, file_name="anomaly-free.csv")
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "skab")
        self.raw_path_train = os.path.join(root, "anomaly-free", "anomaly-free.csv")
        self.raw_paths_test = [os.path.join(root, folder) for folder in ["other", "valve1", "valve2"]]
        self.seed = seed
        self.verbose = verbose
        self.causes = None


    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, index_col='datetime',sep=';',parse_dates=True)
        print("train df len {}".format(len(train_df)))

        test_df = []
        orig_test_len = 0
        for path_name in self.raw_paths_test:
            path_files = os.listdir(path_name)
            for file_name in path_files:
                file_path = os.path.join(path_name, file_name)
                df = pd.read_csv(file_path, index_col='datetime',sep=';', parse_dates=True)
                # print("df of length {} clipped to {}".format(len(df), 100*(len(df)//100)))
                orig_test_len += len(df)
                df = df.head(100*(len(df)//100))
                test_df.append(df)
        test_df = pd.concat(test_df, ignore_index=True, axis=0)
        print("orig test len {}, modified len {}".format(orig_test_len, len(test_df)))
        train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})
        train_df["y"] = np.zeros(len(train_df))
        test_df["y"] = test_df["anomaly"]
        test_df = test_df.drop(columns=["anomaly", "changepoint"], axis=1)
        print(format(test_df.columns))

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, verbose=self.verbose)

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    ds = Skab(seed=seed)
    x_train, y_train, x_test, y_test = ds.data()
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    model = AutoEncoder(sequence_length=100, num_epochs=1, hidden_size=15, lr=1e-4, gpu=None)
    model.fit(x_train)
    error = model.predict(x_test)["error_tc"]
    print(error.shape)


if __name__ == '__main__':
    main()
