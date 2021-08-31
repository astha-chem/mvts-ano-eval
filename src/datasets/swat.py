import math
import pandas as pd
import numpy as np
import os
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
"""
@author: Astha Garg 10/19
"""


class Swat(Dataset):

    def __init__(self, seed: int, shorten_long=True, remove_unique=False, entity=None, verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        if shorten_long:
            name = "swat"
        else:
            name = "swat-long"
        super().__init__(name=name, file_name="SWaT_Dataset_Normal_v1.csv")
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data", "raw", "swat", "raw")
        self.raw_path_train = os.path.join(root, "SWaT_Dataset_Normal_v1.csv")
        self.raw_path_test = os.path.join(root, "SWaT_Dataset_Attack_v0.csv")

        if not os.path.isfile(self.raw_path_train):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Normal_v1.xlsx"))
            df.to_csv(self.raw_path_train, index=False)
        if not os.path.isfile(self.raw_path_test):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Attack_v0.xlsx"))
            df.to_csv(self.raw_path_test, index=False)

        self.seed = seed
        self.shorten_long = shorten_long
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        test_df: pd.DataFrame = pd.read_csv(self.raw_path_test)
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train)

        train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})

        train_df["y"] = train_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        train_df = train_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)
        test_df["y"] = test_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        test_df = test_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)

        # one-hot-encoding stuff
        if self.one_hot:
            keywords = {col_name: "".join([s for s in col_name if not s.isdigit()]) for col_name in train_df.columns}
            cat_cols = [col for col in keywords.keys() if keywords[col] in ["P", "MV", "UV"]]
            one_hot_cols = [col for col in cat_cols if train_df[col].nunique() >= 3 or test_df[col].nunique() >= 3]
            print(one_hot_cols)
            one_hot_encoded = Dataset.one_hot_encoding(pd.concat([train_df, test_df], axis=0, join="inner"),
                                                       col_names=one_hot_cols)
            train_df = one_hot_encoded.iloc[:len(train_df)]
            test_df = one_hot_encoded.iloc[len(train_df):]

        # shorten the extra long anomaly to 550 points
        if self.shorten_long:
            long_anom_start = 227828
            long_anom_end = 263727
            test_df = test_df.drop(test_df.loc[(long_anom_start + 551):(long_anom_end + 1)].index,
                                   axis=0).reset_index(drop=True)
        causes_channels_names = [["MV101"], ["P102"], ["LIT101"], [], ["AIT202"], ["LIT301"], ["DPIT301"],
                                 ["FIT401"], [], ["MV304"], ["MV303"], ["LIT301"], ["MV303"], ["AIT504"],
                                 ["AIT504"], ["MV101", "LIT101"], ["UV401", "AIT502", "P501"], ["P602", "DPIT301",
                                                                                                "MV302"],
                                 ["P203", "P205"], ["LIT401", "P401"], ["P101", "LIT301"], ["P302", "LIT401"],
                                 ["P201", "P203", "P205"], ["LIT101", "P101", "MV201"], ["LIT401"], ["LIT301"],
                                 ["LIT101"], ["P101"], ["P101", "P102"], ["LIT101"], ["P501", "FIT502"],
                                 ["AIT402", "AIT502"], ["FIT401", "AIT502"], ["FIT401"], ["LIT301"]]
        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)

        matching_col_names = np.array([col.split("_1hot")[0] for col in train_df.columns])
        self.causes = []
        for event in causes_channels_names:
            event_causes = []
            for chan_name in event:
                event_causes.extend(np.argwhere(chan_name == matching_col_names).ravel())
            self.causes.append(event_causes)

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    swat = Swat(seed=seed, remove_unique=False, shorten_long=False)
    x_train, y_train, x_test, y_test = swat.data()
    model = AutoEncoder(sequence_length=30, num_epochs=5, hidden_size=15, lr=1e-4, gpu=0)
    model.fit(x_train)
    error = model.predict(x_test)
    print(roc_auc_score(y_test, error))  # e.g. 0.8614


if __name__ == '__main__':
    main()
