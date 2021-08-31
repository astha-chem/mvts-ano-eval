import math
import pandas as pd
import numpy as np
import os
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
"""
@author: Astha Garg 10/19
"""


class Damadics(Dataset):

    def __init__(self, seed: int, remove_unique=False, entity=None, verbose=False, drop_init_test=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        if drop_init_test:
            name = "damadics-s"
        else:
            name = "damadics"
        super().__init__(name=name, file_name="31102001.txt")
        train_filenames = ["31102001.txt"] + ["0"+str(i) + "112001.txt" for i in range(1, 9)]
        test_filenames = ["09112001.txt", "17112001.txt", "20112001.txt"]
        self.test_dates = [date[:-4] for date in test_filenames] # will be used to add labels to dataframes
        self.raw_paths_train = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "damadics", "raw", filename) for filename in train_filenames]
        self.raw_paths_test = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "damadics", "raw", filename) for filename in test_filenames]
        self.anomalies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "damadics", "raw", "DAMADICS_anomalies.csv")
        self.seed = seed
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.drop_init_test = drop_init_test

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_dataframes = [pd.read_csv(path, header=None, sep="\t") for path in self.raw_paths_train]
        test_dataframes = [pd.read_csv(path, header=None, sep="\t") for path in self.raw_paths_test]

        # Adding anomaly labels as a column in the dataframes
        ano_df = pd.read_csv(self.anomalies_path, header=0, dtype=str)
        for df in test_dataframes:
            df["y"] = np.zeros(df.shape[0])
        for i in range(ano_df.shape[0]):
            ano = ano_df.iloc[i, :][["Start_time", "End_time", "Date"]]
            date = ano["Date"]
            df_idx = self.test_dates.index(date)
            start_row = int(ano["Start_time"])
            end_row = int(ano["End_time"])
            test_dataframes[df_idx]["y"].iloc[start_row:(end_row + 1)] = np.ones(1 + end_row - start_row)
        train_df: pd.DataFrame = pd.concat(train_dataframes, axis=0, ignore_index=True)
        test_df: pd.DataFrame = pd.concat(test_dataframes, axis=0, ignore_index=True)
        train_df["y"] = np.zeros(train_df.shape[0])
        # Removing the timestamp from the features
        self.transitions = np.argwhere(test_df.iloc[:, 0] == 0)[1:].ravel()
        train_df = train_df.iloc[:, 1:]
        test_df = test_df.iloc[:, 1:]

        # Removing the beginning of the training set since it seems highly unstable compared to the rest of the sequence
        train_df = train_df.iloc[270000:, :].reset_index(drop=True)
        # Removing the end of the last as we're not sure the last anomaly should be that large
        test_df = test_df.iloc[:-41398, :].reset_index(drop=True)
        if self.drop_init_test:
            test_df = test_df.iloc[45000:, :].reset_index(drop=True)

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)
        self._data = tuple([X_train, y_train, X_test, y_test])

        self.causes = [[4, 5, 6], [4, 5, 6], [4, 5, 6], [26, 27, 28], [26, 27, 28], [16, 17], [16, 17], [4, 5, 6],
                       [19, 20, 21], [19, 20, 21], [4, 5, 6], [19, 20, 21], [25, 26, 27, 28], [25, 26, 27, 28],
                       [25, 26, 27, 28], [1, 2, 3], [16]]

    def get_root_causes(self):
        return self.causes

    def get_transitions(self):
        return self.transitions


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    damadics = Damadics(seed=seed, drop_init_test=True)
    x_train, y_train, x_test, y_test = damadics.data()
    model = AutoEncoder(sequence_length=30, num_epochs=5, hidden_size=15, lr=1e-4, gpu=0)
    model.fit(x_train)
    error = model.predict(x_test)
    print(roc_auc_score(y_test, error))  # e.g. 0.8614


if __name__ == '__main__':
    main()