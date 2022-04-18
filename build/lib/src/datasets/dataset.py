import abc
import os
import pickle
import logging

import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, name: str, file_name: str, entity: str=None):
        self.name = name
        self.processed_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           '../../data/processed/', file_name))

        self._data = None
        self.logger = logging.getLogger(__name__)
        self.train_starts = np.array([])
        self.test_starts = np.array([])
        if entity is None:
            entity = self.name
        self.entity = entity
        self.verbose = False
        self.test_anom_frac_entity = None
        self.test_anom_frac_avg = None
        self.y_test = None

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        """Load data"""

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def get_anom_frac_avg(self):
        """
        hard-coded for multi-entity datasets.
        :return:
        """
        avg_anom_fracs = {"msl": 0.12022773929225528,
                          "smap": 0.12424615960820298,
                          "smd": 0.042119365137533775}
        if self.name in ["swat", "swat-long", "damadics", "damadics-s", "wadi"]:
            test_anom_frac_avg = self.get_anom_frac_entity()
        else:
            me_name = self.name.split("-")[0]
            test_anom_frac_avg = avg_anom_fracs[me_name]
        return test_anom_frac_avg

    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data

    def save(self):
        pickle.dump(self._data, open(self.processed_path, 'wb'))

    def get_dist_based_causes(self, topk=5):
        X_train, _, X_test, y_test = self.data()
        anomalous_events = get_events(y_test)
        n_events = len(anomalous_events.keys())
        visible_causes = []
        for evt_num in range(1, n_events + 1):
            start, end = anomalous_events[evt_num]
            win_size = end - start
            anomalous_seq = X_test.values[start:end, :]
            channel_score = []
            for channel in range(X_test.shape[1]):
                matrix_profile = stumpy.stomp(X_train.values[:, channel], win_size, anomalous_seq[:, channel], False)
                channel_score.append(matrix_profile[:, 0].item())
            ranking_top5 = np.argsort(channel_score)[::-1][:topk]
            visible_causes.append(ranking_top5)
        return visible_causes

    @staticmethod
    def standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
        mini, maxi = X_train.min(), X_train.max()
        for col in X_train.columns:
            if maxi[col] != mini[col]:
                X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
                X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
                X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
            else:
                assert X_train[col].nunique() == 1
                if remove:
                    if verbose:
                        print("Column {} has the same min and max value in train. Will remove this column".format(col))
                    X_train = X_train.drop(col, axis=1)
                    X_test = X_test.drop(col, axis=1)
                else:
                    if verbose:
                        print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                    if mini[col] != 0:
                        X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                        X_test[col] = X_test[col] / mini[col]
                    if verbose:
                        print("After transformation, train unique vals: {}, test unique vals: {}".format(
                        X_train[col].unique(),
                        X_test[col].unique()))
        return X_train, X_test

    def format_data(self, train_df, test_df, OUTLIER_CLASS=1, verbose=False):
        train_only_cols = set(train_df.columns).difference(set(test_df.columns))
        if verbose:
            print("Columns {} present only in the training set, removing them")
        train_df = train_df.drop(train_only_cols, axis=1)

        test_only_cols = set(test_df.columns).difference(set(train_df.columns))
        if verbose:
            print("Columns {} present only in the test set, removing them")
        test_df = test_df.drop(test_only_cols, axis=1)

        train_anomalies = train_df[train_df["y"] == OUTLIER_CLASS]
        test_anomalies: pd.DataFrame = test_df[test_df["y"] == OUTLIER_CLASS]
        print("Total Number of anomalies in train set = {}".format(len(train_anomalies)))
        print("Total Number of anomalies in test set = {}".format(len(test_anomalies)))
        print("% of anomalies in the test set = {}".format(len(test_anomalies) / len(test_df) * 100))
        print("number of anomalous events = {}".format(len(get_events(y_test=test_df["y"].values))))
        # Remove the labels from the data
        X_train = train_df.drop(["y"], axis=1)
        y_train = train_df["y"]
        X_test = test_df.drop(["y"], axis=1)
        y_test = test_df["y"]
        self.y_test = y_test
        return X_train, y_train, X_test, y_test

    @staticmethod
    def one_hot_encoding(df, col_names):
        to_concat = [df]
        for col_name in col_names:
            with_dummies = pd.get_dummies(df[col_name], drop_first=True)
            with_dummies = with_dummies.rename(columns={name: col_name + "_1hot" + str(name) for name in
                                                        with_dummies.columns})
            to_concat.append(with_dummies)
        new_df = pd.concat(to_concat, axis=1).drop(col_names, axis=1)
        return new_df


def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

