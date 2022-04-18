import logging
import sys, os

from src.evaluation.logger_config import init_logging

sys.path.append(os.path.expanduser('~/Projects/multivar-ts-ano/'))

import pickle
import traceback
import copy
import time
from typing import List
import json
import numpy as np
import pandas as pd
from src.algorithms.algorithm_utils import save_torch_algo
from src.datasets import MultiEntityDataset
from src.evaluation.evaluation_utils import fit_distributions, get_scores_channelwise

class Trainer:
    def __init__(self, ds_class, algo_class, algo_config_base: dict, output_dir: str, ds_seed: int=42,
                 ds_kwargs: dict=None, algo_seeds: List[int]=[42], runs_per_seed: int=1, logger=None):
        """
        1 trainer per MultiEntityDataset.
        :param ds_class: Just the class without arguments. eg. Swat
        :param algo_class: eg. AutoEncoder.
        :param ds_seed
        :param ds_kwargs: dictionary of arguments for dataset, aside from seed. eg: {"shorten_long"=False} for swat
        :param algo_config_base: base configuration.
        :param output_dir: results directory where results will be saved in a predecided folder structure.
        :param algo_seeds: this will be passed to the algo along with algo_config_base. If seed is provided in algo_
        config_base, this value will take precedence.
        :param runs_per_seed: for doing multiple runs per seed. In general, this will be 1.
        """
        self.ds_class = ds_class
        self.algo_class = algo_class
        self.ds_seed = ds_seed
        self.ds_kwargs = ds_kwargs
        self.algo_config_base = algo_config_base
        self.train_summary = []
        self.output_dir = output_dir
        self.seeds = algo_seeds
        self.runs_per_seed = runs_per_seed
        self.ds_multi = MultiEntityDataset(dataset_class=ds_class, seed=ds_seed, ds_kwargs=ds_kwargs)
        self.ds_multi_name = self.ds_multi.name
        self.algo_name = self.algo_class(**self.algo_config_base).name
        self.algo_dir = os.path.join(self.output_dir, self.ds_multi_name, self.algo_name)
        os.makedirs(self.algo_dir, exist_ok=True)

        if logger is None:
            init_logging(os.path.join(self.output_dir, 'logs'), prefix="trainer")
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @staticmethod
    def timestamp():
        return time.strftime('%Y-%m-%d-%H%M%S')

    @staticmethod
    def predict(algo, entity, entity_dir, logger):
        X_train, _, X_test, _ = entity.data()
        try:
            algo.batch_size = 4*algo.batch_size
        except:
            logger.warning("couldn't increase batch_size for predict")
        try:
            logger.info("predicting train")
            train_predictions = algo.predict(X_train.copy())
        except Exception as e:
            logger.error(f"encountered exception {e} while predicting train. Will fill with zeros")
            logger.error(traceback.format_exc())
            train_predictions = {'score_t': np.zeros(X_train.shape[0]),
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': None,
                           'recons_tc': None,
                           }
        try:
            logger.info("predicting test")
            test_predictions = algo.predict(X_test.copy())
        except Exception as e:
            logger.error(
                f"encountered exception {e} while predicting test without starts argument. Will fill with zeros")
            logger.error(traceback.format_exc())
            test_predictions = {'score_t': np.zeros(X_test.shape[0]),
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': None,
                           'recons_tc': None,
                           }

        if algo.name == 'OmniAnoAlgo':
            algo.close_session()

        # Put train and test predictions in the same dictionary
        predictions_dic = {}
        for key, value in train_predictions.items():
            predictions_dic[key + "_train"] = value
        for key, value in test_predictions.items():
            predictions_dic[key + "_test"] = value

        predictions_dic["val_recons_err"] = algo.get_val_err()
        predictions_dic["val_loss"] = algo.get_val_loss()

        # When raw errors are available, fit univar gaussians to them to obtain channelwise and time wise anomaly scores
        if predictions_dic["error_tc_test"] is not None and predictions_dic["score_tc_test"] is None and \
                predictions_dic["score_t_test"] is None:
            # go from error_tc to score_tc and score_t using utils functions
            distr_names = ["univar_gaussian"]
            distr_par_file = os.path.join(entity_dir, "distr_parameters")
            distr_params = fit_distributions(distr_par_file, distr_names, predictions_dic=
            {"train_raw_scores": predictions_dic["error_tc_train"]})[distr_names[0]]
            score_t_train, _, score_t_test, score_tc_train, _, score_tc_test = \
                get_scores_channelwise(distr_params, train_raw_scores=predictions_dic["error_tc_train"],
                                       val_raw_scores=None, test_raw_scores=predictions_dic["error_tc_test"],
                                       drop_set=set([]), logcdf=True)
            predictions_dic["score_t_train"] = score_t_train
            predictions_dic["score_tc_train"] = score_tc_train
            predictions_dic["score_t_test"] = score_t_test
            predictions_dic["score_tc_test"] = score_tc_test

        with open(os.path.join(entity_dir, "raw_predictions"), "wb") as file:
            pickle.dump(predictions_dic, file)

        print("Saved predictions_dic")

        return predictions_dic

    def train_predict(self, algo_config=None, config_name="config"):
        """
        :param algo_config:
        :param config_name:
        :return:
        """
        if algo_config is None:
            algo_config = self.algo_config_base
            if config_name is "config":
                config_name = "base-config"
        config_name = config_name.replace("_", "-") + "_" + self.timestamp()
        config_dir = os.path.join(self.algo_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, 'config.json'), 'w') as file:
            json.dump(algo_config, file)

        for seed in self.seeds:
            algo_config["seed"] = seed

            for run_num in range(self.runs_per_seed):
                run_dir = os.path.join(config_dir, str(seed) + "-run" + str(run_num) + "_" + self.timestamp())
                os.makedirs(run_dir)
                self.logger.info("Will train models for {} entities".format(self.ds_multi.num_entities))

                for entity in self.ds_multi.datasets:
                    entity_dir = os.path.join(run_dir, entity.name)
                    os.makedirs(entity_dir)
                    # if self.algo_class == TelemanomAlgo:
                    #     algo_config["entity_id"] = entity.name.split("-", 1)[-1]
                    algo = self.algo_class(**algo_config)
                    algo.set_output_dir(entity_dir)
                    self.logger.info("Training algo {} on entity {} of me_dataset {} with config {}, algo seed {}, "
                                     "run_num {}".format(
                        algo.name, entity.name, self.ds_multi_name, config_name, str(seed), str(run_num)
                    ))
                    X_train, y_train, X_test, y_test = entity.data()
                    
                    try:
                        algo.fit(X_train.copy())
                        train_summary = [config_name, algo_config, algo.get_val_loss()]
                        self.train_summary.append(train_summary)
                        self.logger.info("config {} : {}, val loss {}".format(*train_summary))
                    except Exception as e:
                        self.logger.error(f"encountered exception {e} while training or saving loss")
                        self.logger.error(traceback.format_exc())
                        continue

                    if algo.torch_save:
                        try:
                            save_torch_algo(algo, out_dir=entity_dir)
                        except Exception as e:
                            self.logger.error(f"encountered exception {e} while saving model")
                            self.logger.error(traceback.format_exc())
                    try:
                        self.predict(algo=algo, entity=entity, entity_dir=entity_dir, logger=self.logger)

                    except Exception as e:
                        self.logger.error(f"encountered exception {e} while running predictions")
                        self.logger.error(traceback.format_exc())
                        continue

    def train_modified_configs(self, configs: dict):
        """
        :param configs: dict of configs. The key is the config name. The value is a dict that must specify a valid value
         of an input parameter for the algo. Values not specified will be the ones in the self.algo_config_base, and not
        the default value specified by algo
        :return:
        """
        for config_name, config in configs.items():
            merged_config = copy.deepcopy(self.algo_config_base)
            for key, value in config.items():
                merged_config[key] = value
            self.train_predict(algo_config=merged_config, config_name=config_name)

    def get_best_config(self):
        best_config_name, best_config, best_config_loss = None, None, None

        if len(self.train_summary) == 0:
            self.logger.error(f"Train summary not found. Maybe training hasn't been run yet. Call this function after"
                              f"training is done.")
        if len(self.train_summary) == 1:
            best_config_name, best_config, best_config_loss = self.train_summary[0]
            self.train_summary = pd.DataFrame(self.train_summary, columns=["config name", "config", "val_loss"])
        else:
            try:
                self.train_summary = pd.DataFrame(self.train_summary, columns=["config name", "config", "val_loss"])
                best_config_num = np.nanargmin(self.train_summary["val_loss"].values)
                best_config_name, best_config, best_config_loss = self.train_summary.loc[best_config_num]
            except Exception as e:
                self.logger.error(f"encountered exception {e} while finding best config. val_loss may be missing for "
                                  f"all configs")
                self.logger.error(traceback.format_exc())

        self.train_summary.to_csv(os.path.join(self.algo_dir, "train_summary.csv"), index=False)
        return best_config_name, best_config, best_config_loss
