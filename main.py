import logging
import os
from configs import get_best_config
from src.evaluation.evaluator import analyse_from_pkls

from src.evaluation.logger_config import init_logging
from src.evaluation.trainer import Trainer
from configs import datasets_config, thres_methods, get_thres_config
from src.evaluation.evaluation_utils import get_algo_class, get_dataset_class


def run_multi_seeds(out_dir_root, multi_seeds, ds_to_run, algos_to_run, test_run=False):
    for seed in multi_seeds:
        for ds_name in ds_to_run:
            for algo_name in algos_to_run:
                algo_config_dict = get_best_config(algo_name=algo_name, ds_name=ds_name)
                if test_run:
                    if "num_epochs" in algo_config_dict.keys():
                        algo_config_dict["num_epochs"] = 1
                out_dir_algo = os.path.join(out_dir_root, algo_name)
                train_analyse_algo(ds_name=ds_name, algo_name=algo_name, algo_config_dict=algo_config_dict,
                                   out_dir_algo=out_dir_algo, seed=seed)


def train_analyse_algo(ds_name, algo_name, algo_config_dict, out_dir_algo, seed):
    init_logging(os.path.join(out_dir_algo, 'logs'))
    logger = logging.getLogger(__name__)
    if ds_name in datasets_config.keys():
        ds_kwargs = datasets_config[ds_name]
    else:
        ds_kwargs = {}
    trainer = Trainer(ds_class=get_dataset_class(ds_name),
                      algo_seeds=[seed],
                      algo_class=get_algo_class(algo_name),
                      ds_seed=seed,
                      ds_kwargs=ds_kwargs,
                      algo_config_base=algo_config_dict,
                      output_dir=out_dir_algo,
                      logger=logger)
    print(
        "Training algo {} on dataset {} with config {} and seed {}".format(algo_name, ds_name, algo_config_dict, seed))
    trainer.train_predict()
    analyse_from_pkls(results_root=out_dir_algo, thres_methods=thres_methods, eval_root_cause=True, point_adjust=False,
                      eval_dyn=True, eval_R_model=True, thres_config=get_thres_config,
                      telem_only=True, composite_best_f1=True)


def run_all_benchmarks(out_dir_root):
    multi_seeds = [0, 1, 2, 3, 4]
    ds_to_run = ["swat", "damadics-s", "wadi", "msl", "smap", "smd", "skab"]
    algos_to_run = [
                    "RawSignalBaseline",
                    "PcaRecons",
                    "UnivarAutoEncoder_recon_all",
                    "AutoEncoder_recon_all",
                    "LSTM-ED_recon_all",
                    "TcnED",
                    "VAE-LSTM",
                    "MSCRED",
                    "OmniAnoAlgo"
                    ]
    run_multi_seeds(out_dir_root=out_dir_root,
                    multi_seeds=multi_seeds,
                    ds_to_run=ds_to_run,
                    algos_to_run=algos_to_run,
                    test_run=False)


def run_quick_trial_5_ds(out_dir_root):
    multi_seeds = [0]
    ds_to_run = [
                "damadics-s",
                 "msl",
                 "smap",
                 "smd",
                 "skab"
                 ]
    algos_to_run = ["RawSignalBaseline"]
    run_multi_seeds(out_dir_root=out_dir_root,
                    multi_seeds=multi_seeds,
                    ds_to_run=ds_to_run,
                    algos_to_run=algos_to_run,
                    test_run=True)


def run_quick_trial_all_algos(out_dir_root):
    multi_seeds = [0]
    ds_to_run = ["skab"]
    algos_to_run = [
                    "RawSignalBaseline",
                    "PcaRecons",
                    "UnivarAutoEncoder_recon_all",
                    "AutoEncoder_recon_all",
                    "LSTM-ED_recon_all",
                    "TcnED",
                    "VAE-LSTM",
                    "MSCRED",
                    "OmniAnoAlgo"
                    ]
    run_multi_seeds(out_dir_root=out_dir_root,
                    multi_seeds=multi_seeds,
                    ds_to_run=ds_to_run,
                    algos_to_run=algos_to_run,
                    test_run=True)


if __name__ == "__main__":
    out_dir_root = os.path.join(os.getcwd(), "reports", "trial")
    run_quick_trial_all_algos(out_dir_root)
    # run_quick_trial_5_ds(out_dir_root)
    # run_all_benchmarks(out_dir_root)

