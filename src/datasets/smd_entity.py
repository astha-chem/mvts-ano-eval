import math
import pandas as pd
import numpy as np
import os
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
from src.datasets.dataset import get_events

class Smd_entity(Dataset):

    def __init__(self, seed: int, entity="machine-1-1", remove_unique=False, verbose=False):
        """
        :param seed: for repeatability
        """
        name = "smd-" + entity
        super().__init__(name=name, file_name="machine-1-1.txt")
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "data", "raw", "ServerMachineDataset")
        self.seed = seed
        self.remove_unique = remove_unique
        self.entity = entity
        self.verbose = verbose

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_df = pd.read_csv(os.path.join(self.base_path, "train", self.entity + ".txt"), header=None, sep=",",
                               dtype=np.float32)
        test_df = pd.read_csv(os.path.join(self.base_path, "test", self.entity + ".txt"), header=None, sep=",",
                              dtype=np.float32)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.genfromtxt(os.path.join(self.base_path, "test_label", self.entity + ".txt"), dtype=np.float32,
                                    delimiter=',')
        test_df["y"] = test_labels

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)

        # Retrieve for each anomalous sequence the set of root causes of the anomaly
        self.causes = []
        causes_df = pd.read_csv(os.path.join(self.base_path, "interpretation_label", self.entity + ".txt"), header=None,
                                sep=":", names=["duration", "causes"])
        causes_df["starts"] = causes_df["duration"].str.split("-").map(lambda x: int(x[0]))
        events_df = pd.DataFrame(list(get_events(y_test=y_test).values()), columns=["starts", "ends"])
        merged_df = pd.DataFrame.merge(events_df, causes_df, how="outer", left_on="starts", right_on="starts")\
            .sort_values(by="starts")
        starts_with_missing_causes = merged_df["starts"][pd.isna(merged_df["causes"])]
        if len(starts_with_missing_causes) > 0:
            print("Events starting at {} don't have root causes. Will be filled with all channels as root cause".format(
                starts_with_missing_causes.values))
        # if an event was present but root causes not provided, assign all channels to true root cause
        merged_df["causes"] = merged_df["causes"].fillna(str(list(range(1, X_test.shape[1]+1))).replace(" ", "").replace(
            "[", "").replace("]", ""))
        for row in merged_df["causes"]:
            event_causes = [int(cause) - 1 for cause in row.split(",")]
            self.causes.append(event_causes)

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    smd = Smd_entity(seed=seed, remove_unique=False, entity="machine-3-3")
    x_train, y_train, x_test, y_test = smd.data()
    model = AutoEncoder(sequence_length=30, num_epochs=2, hidden_size=15, lr=1e-4, gpu=0)
    model.fit(x_train)
    error = model.predict(x_test)
    print(roc_auc_score(y_test, error))  # e.g. 0.8614


if __name__ == '__main__':
    main()
