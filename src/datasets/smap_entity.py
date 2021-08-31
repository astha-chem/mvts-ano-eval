import math
import pandas as pd
import numpy as np
import os
import ast
from src.datasets import Dataset
from sklearn.metrics import roc_auc_score
from src.datasets.entities_names import smap_entities


class Smap_entity(Dataset):

    def __init__(self, seed: int, entity="A-1", remove_unique=False, verbose=False):
        """
        :param seed: for repeatability
        """
        if entity in smap_entities:
            name = "smap-" + entity
        else:
            name = "not_found"
            print("Entity name not recognized")
        super().__init__(name=name, file_name="A-1.npy")
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "data", "raw", "smap_msl")
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
        train_values = np.load(os.path.join(self.base_path, "train", self.entity + ".npy"))
        train_df = pd.DataFrame(train_values)
        test_values = np.load(os.path.join(self.base_path, "test", self.entity + ".npy"))
        test_df = pd.DataFrame(test_values)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.zeros(test_df.shape[0])
        labels_df = pd.read_csv(os.path.join(self.base_path, "labeled_anomalies.csv"), header=0, sep=",")
        entity_attacks = labels_df[labels_df["chan_id"] == self.entity]["anomaly_sequences"].values
        entity_attacks = ast.literal_eval(entity_attacks[0])
        for sequence in entity_attacks:
            test_labels[sequence[0]:(sequence[1] + 1)] = 1
        test_df["y"] = test_labels

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)

        # We don't have any root cause labels for this data set
        self.causes = None

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def main():
    from src.algorithms import AutoEncoder
    seed = 0
    smap = Smap_entity(seed=seed, remove_unique=False)
    x_train, y_train, x_test, y_test = smap.data()
    model = AutoEncoder(sequence_length=30, num_epochs=2, hidden_size=15, lr=1e-4, gpu=0)
    model.fit(x_train)
    error = model.predict(x_test)
    print(roc_auc_score(y_test, error))  # e.g. 0.8614


if __name__ == '__main__':
    main()
