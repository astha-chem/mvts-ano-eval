import math
import pandas as pd
import numpy as np
import os
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
"""
@author: Astha Garg 10/19
"""


class Wadi(Dataset):

    def __init__(self, seed: int, remove_unique=False, entity=None, verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        super().__init__(name="wadi", file_name="WADI_14days.csv")
        self.raw_path_train = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "wadi", "raw", "WADI_14days.csv")
        self.raw_path_test = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          "data", "raw", "wadi", "raw", "WADI_attackdata.csv")
        self.anomalies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "wadi", "raw", "WADI_anomalies.csv")
        self.seed = seed
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        test_df: pd.DataFrame = pd.read_csv(self.raw_path_test, header=0)
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, header=3)

        # Removing 4 columns who only contain nans (data missing from the csv file)
        nan_columns = [r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS']
        train_df = train_df.drop(nan_columns, axis=1)
        test_df = test_df.drop(nan_columns, axis=1)

        train_df = train_df.rename(columns={col: col.split('\\')[-1] for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.split('\\')[-1] for col in test_df.columns})

        # Adding anomaly labels as a column in the dataframes
        ano_df = pd.read_csv(self.anomalies_path, header=0)
        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = np.zeros(test_df.shape[0])
        causes = []
        for i in range(ano_df.shape[0]):
            ano = ano_df.iloc[i, :][["Start_time", "End_time", "Date"]]
            start_row = np.where((test_df["Time"].values == ano["Start_time"]) &
                                 (test_df["Date"].values == ano["Date"]))[0][0]
            end_row = np.where((test_df["Time"].values == ano["End_time"]) &
                               (test_df["Date"].values == ano["Date"]))[0][0]
            test_df["y"].iloc[start_row:(end_row + 1)] = np.ones(1 + end_row - start_row)
            causes.append(ano_df.iloc[i, :]["Causes"])
        # Removing time and date from features
        train_df = train_df.drop(["Time", "Date", "Row"], axis=1)
        test_df = test_df.drop(["Time", "Date", "Row"], axis=1)

        if self.one_hot:
            # actuator colums (categoricals) with < 2 categories (all of these have 3 categories)
            one_hot_cols = ['1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS', '2_MV_003_STATUS',
                            '2_MV_006_STATUS', '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS',
                            '2_MV_501_STATUS', '2_MV_601_STATUS']

            # combining before encoding because some categories only seen in test
            one_hot_encoded = Dataset.one_hot_encoding(pd.concat([train_df, test_df], axis=0, join="inner"),
                                                       col_names=one_hot_cols)
            train_df = one_hot_encoded.iloc[:len(train_df)]
            test_df = one_hot_encoded.iloc[len(train_df):]

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)

        matching_col_names = np.array([col.split("_1hot")[0] for col in train_df.columns])
        self.causes = []
        for event in causes:
            event_causes = []
            for chan_name in get_chan_name(event):
                event_causes.extend(np.argwhere(chan_name == matching_col_names).ravel())
            self.causes.append(event_causes)

        self.visible_causes = [[5, 6, 10, 14, 16, "105"], [6, "9", "13", 14, 16, "18", "21", "40", 49, 55, 61, 68, 102],
                               [1, 5, 6, "9", 14, 16, 23, 25, 26, 29, 35, "37", 43, 47, 51, 63, 67], [24, 27, 30, 33,
                                                                                                      36, 40, 63, 64,
                                                                                                      65, 66, 67, 68,
                                                                                                      82, 84, 86, 87,
                                                                                                      89, 90, 91, 104],
                               [9, 18, 20, 22, 25, 39, 40, 42, 43, "61", 86, 87, 89, 90, 91, 102], [1, 2, 3, 6, 11, 12,
                                                                                                    14, 16, 23, 26, 29,
                                                                                                    32, 35, 38],
                               [62, 110], [19, 39], [1, 2, 3, 4, 5, 6, 10, 14, 16, 18, 39, 71], ["18", "39", 62, "71",
                                                                                                 86, 100, 110, 111],
                               [29, 32, 35, 38, "40", 62, 98, 99, 110, 111], [88, 123], [14, 16, 33, 56, 110, 111],
                               [1, 3, 6, 14, 16, 61, 101, 103]]

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def get_chan_name(chan_list_str):
    if len(chan_list_str) > 2:
        chan_list_str = chan_list_str[2:-2]
        chan_list = chan_list_str.split("', '")
        return chan_list
    else:
        return []


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    wadi = Wadi(seed=seed)
    x_train, y_train, x_test, y_test = wadi.data()
    print(wadi.causes)
    # model = AutoEncoder(sequence_length=30, num_epochs=5, hidden_size=15, lr=1e-4, gpu=0)
    # model.fit(x_train)
    # error = model.predict(x_test)
    # print(roc_auc_score(y_test, error))  # e.g. 0.8614


if __name__ == '__main__':
    main()
