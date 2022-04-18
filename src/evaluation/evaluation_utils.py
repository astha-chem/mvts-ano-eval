"""
run predictions for multiple algos and datasets from saved_model pkl files
@author Astha Garg
Dec 2019
"""

import umap
import traceback

from configs import default_thres_config, constant_std
# from src.algorithms.telem_anom.errors import Errors

# from src.algorithms.telem_anom.channel import Channel
from src.datasets.skab import Skab

from src.datasets.swat import Swat
from src.datasets.wadi import Wadi
from src.datasets.damadics import Damadics
from src.datasets.smap_entity import Smap_entity
from src.datasets.msl_entity import Msl_entity
from src.datasets.smd_entity import Smd_entity
from src.datasets.dataset import get_events
from src.algorithms.algorithm_utils import *
from src.algorithms import UnivarAutoEncoder, AutoEncoder, LSTMED, VAE_LSTM, TcnED, RawSignalBaseline, \
    PcaRecons, MSCRED, OmniAnoAlgo#, Ocsvm_gpu, TelemanomAlgo
from scipy import signal


from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, \
    accuracy_score, fbeta_score, average_precision_score
from scipy.stats import norm
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

datasets_dict = {"msl": Msl_entity,
                 "smap": Smap_entity,
                 "smd": Smd_entity,
                 "damadics": Damadics,
                 "wadi": Wadi,
                 "swat": Swat,
                 "swat-long": Swat,
                 "damadics-s": Damadics,
                 "skab": Skab
                 }

def get_chan_num(abs_filename):
    return int(abs_filename.split("_")[-1])


def get_dataset_class(ds_name):
    return datasets_dict[ds_name]


def get_algo_class(algo_name):
    algos = {"AutoEncoder_recon_all": AutoEncoder,
             "UnivarAutoEncoder_recon_all": UnivarAutoEncoder,
             "LSTM-ED": LSTMED,
             "LSTM-ED_recon_all": LSTMED,
             "PcaRecons": PcaRecons,
             "VAE-LSTM": VAE_LSTM,
             "TcnED": TcnED,
             "TelemanomAlgo": TelemanomAlgo,
             "RawSignalBaseline": RawSignalBaseline,
             "MSCRED": MSCRED,
             "OmniAnoAlgo": OmniAnoAlgo,
             "Ocsvm_gpu": Ocsvm_gpu
             }
    assert algo_name in list(algos.keys()), "Algo name {} not identified".format(algo_name)
    return algos[algo_name]


def threshold_and_predict(score_t_test, y_test, true_events, logger,  test_anom_frac, thres_method="top_k_time",
                          point_adjust=False, score_t_train=None, thres_config_dict= dict(), return_auc=False,
                          composite_best_f1=False):
    if thres_method in thres_config_dict.keys():
        config = thres_config_dict[thres_method]
    else:
        config = default_thres_config[thres_method]
    # test_anom_frac = (np.sum(y_test)) / len(y_test)
    auroc = None
    avg_prec = None
    if thres_method == "thresholded_score":
        opt_thres = 0.5
        if set(score_t_test) - {0, 1}:
            logger.error("Score_t_test isn't binary. Predicting all as non-anomalous")
            pred_labels = np.zeros(len(score_t_test))
        else:
            pred_labels = score_t_test

    elif thres_method == "best_f1_test" and point_adjust:
        prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
        if not config["exact_pt_adj"]:
            fscore_best_time = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore_best_time))
            opt_thres = thresholds[opt_num]
            thresholds = np.random.choice(thresholds, size=5000) + [opt_thres]
        fscores = []
        for thres in thresholds:
            _, _, _, _, _, fscore = get_point_adjust_scores(y_test, score_t_test > thres, true_events)
            fscores.append(fscore)
        opt_thres = thresholds[np.argmax(fscores)]
        pred_labels = score_t_test > opt_thres

    elif thres_method == "best_f1_test" and composite_best_f1:
        prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
        precs_t = prec
        fscores_c = [get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t) for thres, prec_t in
                     zip(thresholds, precs_t)]
        try:
            opt_thres = thresholds[np.nanargmax(fscores_c)]
        except:
            opt_thres = 0.0
        pred_labels = score_t_test > opt_thres

    elif thres_method == "top_k_time":
        opt_thres = np.nanpercentile(score_t_test, 100 * (1 - test_anom_frac), interpolation='higher')
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "best_f1_test":
        prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
        fscore = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
        opt_num = np.squeeze(np.argmax(fscore))
        opt_thres = thres[opt_num]
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif "tail_prob" in thres_method:
        tail_neg_log_prob = config["tail_prob"]
        opt_thres = tail_neg_log_prob
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "nasa_npt":
        opt_thres = 0.5
        pred_labels = get_npt_labels(score_t_test, y_test, config)
    else:
        logger.error("Thresholding method {} not in [top_k_time, best_f1_test, tail_prob]".format(thres_method))
        return None, None
    if return_auc:
        avg_prec = average_precision_score(y_test, score_t_test)
        auroc = roc_auc_score(y_test, score_t_test)
        return opt_thres, pred_labels, avg_prec, auroc
    return opt_thres, pred_labels


def get_point_adjust_scores(y_test, pred_labels, true_events):
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if pred_labels[true_start:true_end].sum() > 0:
            tp += (true_end - true_start)
        else:
            fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore


def evaluate_predicted_labels(pred_labels, y_test, true_events, logger, eval_method="time-wise", breaks=[],
                              point_adjust=False):
    r"""
    Computes evaluation metrics for the binary classifications given the true and predicted labels
    :param point_adjust:
    :param pred_labels: array of predicted labels
    :param y_test: array of true labels
    :param eval_method: string that indicates whether we evaluate the classification time point-wise or event-wise
    :param breaks: array of discontinuities in the time series, relevant only if you look at event-wise
    :param return_raw: Boolean that indicates whether we want to return tp, fp and fn or prec, recall and f1
    :return: tuple of evaluation metrics
    """

    if eval_method == "time-wise":
        if point_adjust:
            fp, fn, tp, prec, rec, fscore = get_point_adjust_scores(y_test, pred_labels, true_events)
        else:
            _, prec, rec, fscore, _ = get_accuracy_precision_recall_fscore(y_test, pred_labels)
            tp = np.sum(pred_labels*y_test)
            fp = np.sum(pred_labels) - tp
            fn = np.sum(y_test) - tp

    else:
        logger.error("Evaluation method {} not in [time-wise, event-wise]".format(eval_method))
        return 0, 0, 0

    return tp, fp, fn, prec, rec, fscore


def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
    accuracy = accuracy_score(y_true, y_pred)
    # warn_for=() avoids log warnings for any result being zero
    # precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_score = (2 * precision * recall)/(precision + recall)
    if precision == 0 and recall == 0:
        f05_score = 0
    else:
        f05_score = fbeta_score(y_true, y_pred, average='binary', beta=0.5)
    return accuracy, precision, recall, f_score, f05_score


def get_scores_channelwise(distr_params, train_raw_scores, val_raw_scores, test_raw_scores, drop_set=set([]), logcdf=False):
    use_ch = list(set(range(test_raw_scores.shape[1])) - drop_set)
    train_prob_scores = -1 * np.concatenate(([get_per_channel_probas(train_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                        for i in range(train_raw_scores.shape[1])]), axis=1)
    test_prob_scores = [get_per_channel_probas(test_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                       for i in range(train_raw_scores.shape[1])]
    test_prob_scores = -1 * np.concatenate(test_prob_scores, axis=1)

    train_ano_scores = np.sum(train_prob_scores[:, use_ch], axis=1)
    test_ano_scores = np.sum(test_prob_scores[:, use_ch], axis=1)

    if val_raw_scores is not None:
        val_prob_scores = -1 * np.concatenate(([get_per_channel_probas(val_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                          for i in range(train_raw_scores.shape[1])]), axis=1)
        val_ano_scores = np.sum(val_prob_scores[:, use_ch], axis=1)
    else:
        val_ano_scores = None
        val_prob_scores = None
    return train_ano_scores, val_ano_scores, test_ano_scores, train_prob_scores, val_prob_scores, test_prob_scores


# Computes (when not already saved) parameters for scoring distributions
def fit_distributions(distr_par_file, distr_names, predictions_dic, val_only=False):
    # try:
    #     with open(distr_par_file, 'rb') as file:
    #         distributions_dic = pickle.load(file)
    # except:
    distributions_dic = {}
    for distr_name in distr_names:
        if distr_name in distributions_dic.keys():
            continue
        else:
            print("The distribution parameters for %s for this algorithm on this data set weren't found. \
            Will fit them" % distr_name)
            if "val_raw_scores" in predictions_dic:
                raw_scores = np.concatenate((predictions_dic["train_raw_scores"], predictions_dic["val_raw_scores"]))
            else:
                raw_scores = predictions_dic["train_raw_scores"]

            distributions_dic[distr_name] = [fit_univar_distr(raw_scores[:, i], distr=distr_name)
                                             for i in range(raw_scores.shape[1])]
    with open(distr_par_file, 'wb') as file:
        pickle.dump(distributions_dic, file)

    return distributions_dic


def get_cause_ranking(score_tc_test, y_test, pred_labels=None, tp_only=True):
    r"""
    Return for each anomalous event, the ranking of the channels responsible for the anomaly
    :param score_tc_test: array of scores of shape (n_time_points, n_channels)
    :param y_test: array of true labels
    :param pred_labels: array of predicted labels
    :param tp_only: Boolean to indicate whether to use only time points that were detected as anomalies for the ranking
    of the channels or all the time points labelled as anomalous
    :return: List of rankings, there are n_events lists each of length n_channels
    """
    anomalous_events = get_events(y_test)
    n_events = len(anomalous_events.keys())
    ranking = []
    detected_events = [True]*n_events
    for event in range(1, n_events + 1):
        start, end = anomalous_events[event]
        mask = np.array(score_tc_test.shape[0] * [False])
        mask[start:end] = True

        # using tp only
        if tp_only:
            assert pred_labels is not None, "Predicted labels are required for root cause identification with tp only"
            mask_tp_only = np.logical_and(mask, pred_labels.astype(bool))
            if mask_tp_only.sum() == 0:
                detected_events[event-1] = False
            # rank for all events, even if not detected. But we'll only consider detected events later.
            channel_scores_tp_only = np.sum(score_tc_test[mask_tp_only, :], axis=0).ravel()
            ranking.append(list(np.argsort(-channel_scores_tp_only)))

        # all anomalous time points, without mask
        else:
            channel_scores_all = np.sum(score_tc_test[mask, :], axis=0).ravel()
            ranking.append(list(np.argsort(-channel_scores_all)))

    return ranking, detected_events


def hitrate_cause_eval(event_causes, ranking):
    r"""
    Compute the hit rate metric to evaluate root cause identification on one anomalous event
    :param event_causes: list of true causes for one anomalous event
    :param ranking: ranking predicted by the algorithm for one anomalous event
    :return: Tuple of two elements: hitrate100 and hitrate150
    """
    n_causes = len(event_causes)
    if n_causes == 0:
        return np.nan, np.nan
    ground_truth = set(event_causes)
    hitrate_100 = len(ground_truth.intersection(set(ranking[:n_causes])))/n_causes
    num_causes_150 = min(len(ranking), int(1.5 * n_causes))
    hitrate_150 = len(ground_truth.intersection(set(ranking[:num_causes_150])))/n_causes
    return hitrate_100, hitrate_150


def rca_evaluation(ground_truth, rankings, detected_events: List[bool]):
    r"""
    Wrapping method to evaluate root cause identification on all anomalous events inside a time series
    :param ground_truth: list of true causes for each anomalous event, list that contains n_events sublists
    :param rankings: List of rankings, there are n_events lists each of length n_channels
    :param detected_events: List indicating True or False for each event, indicating whether or not it was detected.
    :return: Two lists for the two metrics, each of length n_events
    """
    n_events = len(rankings)
    if not ground_truth:
        return n_events*[np.nan], n_events*[np.nan]
    hr_100_list = []
    hr_150_list = []
    rc_top3_list = []
    for event in range(n_events):
        if detected_events[event]:
            hr_100, hr_150 = hitrate_cause_eval(ground_truth[event], rankings[event])
            hr_100_list.append(hr_100)
            hr_150_list.append(hr_150)
            if set(ground_truth[event]).intersection(set(rankings[event][:3])):
                rc_top3 = 1
            else:
                rc_top3 = 0
            rc_top3_list.append(rc_top3)

    return hr_100_list, hr_150_list, rc_top3_list


def moving_average(score_t, window=3):
    # return length = len(score_t) - window + 1
    ret = np.cumsum(score_t, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window):
    n_t = error_t_test.shape[0]

    # assuming that length of scores is always greater than short_window
    short_term_means = np.concatenate((error_t_test[:short_window - 1], moving_average(error_t_test, short_window)))
    if long_window >= n_t:
        long_win = n_t - 1
    else:
        long_win = long_window

    if error_t_train is None:
        init_score_t_test = np.zeros(long_win - 1)
        means_test_t = moving_average(error_t_test, long_win)
        stds_test_t = np.array(pd.Series(error_t_test).rolling(window=long_win).std().values)[long_win - 1:]
        stds_test_t[stds_test_t == 0] = constant_std
        distribution = norm(0, 1)
        score_t_test_dyn = -distribution.logsf((short_term_means[(long_win - 1):] - means_test_t) / stds_test_t)
        score_t_test_dyn = np.concatenate([init_score_t_test, score_t_test_dyn])
    else:
        if len(error_t_train) < long_win - 1:
            full_ts = np.concatenate([np.zeros(long_win - 1 - len(error_t_train)), error_t_train, error_t_test], axis=0)
        else:
            full_ts = np.concatenate([error_t_train[-long_win + 1:], error_t_test], axis=0)
        means_test_t = moving_average(full_ts, long_win)
        stds_test_t = np.array(pd.Series(full_ts).rolling(window=long_win).std().values)[long_win - 1:]
        stds_test_t[stds_test_t == 0] = constant_std
        distribution = norm(0, 1)
        score_t_test_dyn = -distribution.logsf((short_term_means - means_test_t) / stds_test_t)

    return score_t_test_dyn


def get_dynamic_scores(error_tc_train, error_tc_test, error_t_train, error_t_test, long_window=2000, short_window=10):
    # if error_tc is available, it will be used rather than error_t
    if error_tc_test is None:
        score_tc_test_dyn = None
        score_tc_train_dyn = None
        score_t_test_dyn = get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window)
        if error_t_train is not None:
            score_t_train_dyn = get_dynamic_score_t(None, error_t_train, long_window, short_window)
        else:
            score_t_train_dyn = None
    else:
        n_cols = error_tc_test.shape[1]
        if error_tc_train is not None:
            score_tc_test_dyn = np.stack([get_dynamic_score_t(error_tc_train[:, col], error_tc_test[:, col],
                                                              long_window, short_window) for col in range(n_cols)],
                                         axis=-1)
            score_tc_train_dyn = np.stack([get_dynamic_score_t(None, error_tc_train[:, col],
                                                                    long_window, short_window) for col in range(n_cols)]
                                          , axis=-1)
            score_t_train_dyn = np.sum(score_tc_train_dyn, axis=1)
        else:
            score_tc_test_dyn = np.stack([get_dynamic_score_t(None, error_tc_test[:, col],
                                                              long_window, short_window) for col in range(n_cols)],
                                         axis=-1)
            score_t_train_dyn = None
            score_tc_train_dyn = None

        score_t_test_dyn = np.sum(score_tc_test_dyn, axis=1)

    return score_t_test_dyn, score_tc_test_dyn, score_t_train_dyn, score_tc_train_dyn


def get_gaussian_kernel_scores(score_t_dyn, score_tc_dyn, kernel_sigma):
    # if error_tc is available, it will be used rather than error_t
    gaussian_kernel = signal.gaussian(kernel_sigma * 8, std=kernel_sigma)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    if score_tc_dyn is None:
        score_tc_dyn_gauss_conv = None
        score_t_dyn_gauss_conv = signal.convolve(score_t_dyn, gaussian_kernel, mode="same")
    else:
        n_cols = score_tc_dyn.shape[1]
        score_tc_dyn_gauss_conv = np.stack([signal.convolve(score_tc_dyn[:, col], gaussian_kernel, mode="same")
                                      for col in range(n_cols)], axis=-1)
        score_t_dyn_gauss_conv = np.sum(score_tc_dyn_gauss_conv, axis=1)

    return score_t_dyn_gauss_conv, score_tc_dyn_gauss_conv


def collect_eval_metrics(algo_results, score_t_test, y_test, thres_methods, rca_possible, true_events, logger,
                         dataset, score_tc_test=None, root_causes=None, score_t_train=None, point_adjust=False,
                         thres_config_dict=dict(), eval_methods=["time-wise"],
                         make_plots=["prc", "score_t"], plots_name="", composite_best_f1=False, score_tc_train=None):
    r"""
    Wrapping method to threshold score, evaluate predictions and store them in a dictionary
    :param algo_results: dictionary used to store the results across all entities
    :param score_t_test: Array of anomaly scores of shape n_time points
    :param y_test: Array of true labels
    :param thres_methods: List of thresholding methods to use for the evaluation
    :param rca_possible: Boolean to indicate whether root cause analysis can and should be performed
    :param true_events: a dictionary about event start and end generated using function get_events
    :param score_tc_test: If rca_possible is True, this is an array of scores of shape n_time_points x n_channels used
    to identify channels responsible for the anomalies
    :param root_causes: List of ground truth causes for each anomalous event
    :return: The results dictionary filled with evaluation metrics
    """
    # Evaluate on each entity
    test_anom_frac_entity = dataset.get_anom_frac_entity()
    for thres_method in thres_methods:
        # all thresholding is done time-wise, eval may be done event-wise
        logger.info("Thresholding with {} method".format(thres_method))
        opt_thres, pred_labels, avg_prec, auroc = threshold_and_predict(score_t_test, y_test, true_events=true_events,
                                                                        logger=logger,
                                                                        test_anom_frac=test_anom_frac_entity,
                                                                        thres_method=thres_method,
                                                                        point_adjust=point_adjust,
                                                                        score_t_train=score_t_train,
                                                                        thres_config_dict=thres_config_dict,
                                                                        return_auc=True,
                                                                        composite_best_f1=composite_best_f1)
        algo_results[thres_method]["opt_thres"].append(opt_thres)
        prec_t, rec_e, fscore_c = get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=True)
        algo_results[thres_method]["fscore_comp"].append(fscore_c)
        algo_results[thres_method]["rec_e"].append(rec_e)

        if avg_prec is not None and auroc is not None:
            algo_results['auroc'].append(auroc)
            algo_results['avg_prec'].append(avg_prec)
        if "score_t" in make_plots:
            x_train, _, x_test, _ = dataset.data()
            plot_scoring_func(y_test, score_t_test, thres_method, opt_thres, pred_labels, plots_name+"_"+thres_method,
                              colorby_anom=False, start_stop=None, label=plots_name.split("_")[-1],
                              x_train=x_train.values, x_test=x_test.values, score_tc_train=score_tc_train,
                              score_tc_test=score_tc_test, plot_channels=True)

        for eval_method in eval_methods:
            logger.info("Eval with {} method".format(eval_method))
            tp, fp, fn, prec, rec, fscore = evaluate_predicted_labels(pred_labels, y_test, logger=logger,
                                                                      true_events=true_events,
                                                                      eval_method=eval_method,
                                                                      point_adjust=point_adjust)
            algo_results[thres_method][eval_method]["tp"].append(tp)
            algo_results[thres_method][eval_method]["fp"].append(fp)
            algo_results[thres_method][eval_method]["fn"].append(fn)
        if rca_possible:
            logger.info("Carrying out root cause analysis tp only")
            ranking_tp, detected_events = get_cause_ranking(score_tc_test, y_test, pred_labels=pred_labels,
                                                            tp_only=True)
            hr_100_tp, hr_150_tp, rc_top3_tp = rca_evaluation(root_causes, ranking_tp, detected_events)
            algo_results[thres_method]["hr_100_tp"].append(hr_100_tp)
            algo_results[thres_method]["hr_150_tp"].append(hr_150_tp)
            algo_results[thres_method]["rc_top3_tp"].append(rc_top3_tp)

    if rca_possible:
        logger.info("Carrying out root cause analysis for all ground truth")
        ranking_all, detected_events = get_cause_ranking(score_tc_test, y_test, pred_labels=None, tp_only=False)
        hr_100_all, hr_150_all, rc_top3_all = rca_evaluation(root_causes, ranking_all, detected_events)
        algo_results["hr_100_all"].append(hr_100_all)
        algo_results["hr_150_all"].append(hr_150_all)
        algo_results["rc_top3_all"].append(rc_top3_all)
    return algo_results


def combine_entities_eval_metrics(algo_results, thres_methods, global_ds_name, algo_name, rca_possible,
                                  eval_methods=["time-wise"]):
    r"""
    Wrapping method to combine results across all entities
    :param algo_results: dictionary where results from each entity are stored
    :param thres_methods: List of thresholding methods used in the evaluation
    :param global_ds_name: Name of the multi entity data set on which the algorithm is evaluated
    :param algo_name: Name of the algorithm being evaluated
    :param rca_possible: Boolean to indicate whether root cause analysis can and should be performed
    :return: A list of results ready to eb converted in a Dataframe where each line is a configuration of evaluation
    """
    # todo: collect the validation loss
    results = []
    auroc = np.nanmean(algo_results["auroc"])
    avg_prec = np.nanmean(algo_results["avg_prec"])
    for thres_method in thres_methods:
        fscore_comp_raw = algo_results[thres_method]["fscore_comp"]
        fscore_comp_avg = np.nanmean(fscore_comp_raw)
        rec_event_raw = algo_results[thres_method]["rec_e"]
        rec_event_avg = np.nanmean(rec_event_raw)
        eval_methods.sort()
        for eval_method in eval_methods:
            prec_overall, rec_overall, fscore_overall = get_prec_rec_fscore(sum(algo_results[thres_method][eval_method]["tp"]),
                                                    sum(algo_results[thres_method][eval_method]["fp"]),
                                                    sum(algo_results[thres_method][eval_method]["fn"]))
            prec_avg = []
            rec_avg = []
            fscore_avg = []
            for tp, fp, fn in zip(algo_results[thres_method][eval_method]["tp"],
                                  algo_results[thres_method][eval_method]["fp"],
                                  algo_results[thres_method][eval_method]["fn"]):
                ent_prec, ent_rec, ent_fscore = get_prec_rec_fscore(tp, fp, fn)
                prec_avg.append(ent_prec)
                rec_avg.append(ent_rec)
                fscore_avg.append(ent_fscore)
            prec_avg = np.average(prec_avg)
            rec_avg = np.average(rec_avg)
            fscore_avg = np.average(fscore_avg)

            if rca_possible:
                # first take mean over each entity, then mean over all entities
                hitrate_100_tp = np.nanmean([np.nanmean(res) for res in algo_results[thres_method]["hr_100_tp"]])
                hitrate_150_tp = np.nanmean([np.nanmean(res) for res in algo_results[thres_method]["hr_150_tp"]])
                hitrate_100_all = np.nanmean([np.nanmean(res) for res in algo_results["hr_100_all"]])
                hitrate_150_all = np.nanmean([np.nanmean(res) for res in algo_results["hr_150_all"]])
                rc_top3_tp = np.nanmean([np.nanmean(res) for res in algo_results[thres_method]["rc_top3_tp"]])
                rc_top3_all = np.nanmean([np.nanmean(res) for res in algo_results["rc_top3_all"]])
            else:
                hitrate_100_tp, hitrate_150_tp, hitrate_100_all, hitrate_150_all, rc_top3_tp, rc_top3_all = \
                    None, None, None, None, None, None

            val_loss_raw = algo_results["val_loss"]
            try:
                val_loss_avg = np.nanmean(val_loss_raw)
            except:
                val_loss_avg = None

            val_recons_err_raw = algo_results["val_recons_err"]

            try:
                val_recons_err_avg = np.nanmean(val_recons_err_raw)
            except:
                val_recons_err_avg = None

            std_scores_train = algo_results["std_scores_train"]

            metrics_to_save = [global_ds_name, algo_name, eval_method, prec_overall, rec_overall, fscore_overall,
                               prec_avg, rec_avg, fscore_avg, rec_event_avg, fscore_comp_avg, thres_method, auroc, avg_prec,
                               algo_results["auroc"],
                               algo_results["avg_prec"], val_loss_avg, val_recons_err_avg,
                               hitrate_100_tp, hitrate_150_tp, hitrate_100_all, hitrate_150_all, rc_top3_tp,
                               rc_top3_all, algo_results[thres_method][eval_method]["tp"],
                               algo_results[thres_method][eval_method]["fp"],
                               algo_results[thres_method][eval_method]["fn"], algo_results[thres_method]["hr_100_tp"],
                               algo_results[thres_method]["hr_150_tp"], algo_results["hr_100_all"],
                               algo_results["hr_150_all"], algo_results[thres_method]["rc_top3_tp"],
                               algo_results["rc_top3_all"], val_loss_raw, val_recons_err_raw, std_scores_train,
                               algo_results[thres_method]["opt_thres"], fscore_comp_raw, rec_event_raw]
            results.append(metrics_to_save)

    column_names = ["dataset", "algo", "eval_method", "prec_overall", "rec_overall", "fscore_overall",
                    "prec_avg", "rec_avg", "fscore_avg", "rec_event_avg", "fscore_comp_avg", "thres_method",
                    "auroc_avg", "avg_prec_avg", "auroc_raw", "avg_prec_raw", "val_loss_avg", "val_recons_err_avg",
                    "hitrate_100_tp", "hitrate_150_tp", "hitrate_100_all", "hitrate_150_all", "rc_top3_tp",
                    "rc_top3_all", "tp_raw", "fp_raw", "fn_raw", "hr_100_tp_raw", "hr_150_tp_raw",
                    "hr_100_all_raw", "hr_150_all_raw", "rc_top3_tp_raw", "rc_top3_all_raw",
                    "val_loss_raw", "val_recons_err_raw", "std_scores_train", "opt_thres_raw", "fscore_comp_raw",
                    "rec_event_raw"]
    return results, column_names


def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore


def plot_prc(y_test, score_t_test, name, save_dir, point_adjust=False):
    # plot and save the precision_recall curve

    if point_adjust:
        saved_name = os.path.join(save_dir, f"prc_pt_adj_{name}.pdf")
        true_events = get_events(y_test)
        thresholds = np.linspace(min(score_t_test), max(score_t_test), 100)
        prec = []
        rec = []
        fps = []
        fns = []
        tps = []
        for thres in thresholds:
            fp, fn, tp, precision, recall, _= get_point_adjust_scores(y_test, score_t_test > thres, true_events)
            prec.append(precision)
            rec.append(recall)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
    else:
        saved_name = os.path.join(save_dir, f"prc_{name}.pdf")
        prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(prec) <= 20000:
        ax.plot(rec, prec, 'o', label="Precision-recall-curve")
    else:
        ax.plot(rec, prec, label="Precision-recall-curve")
    ax.set_title(name)
    # todo: the following doesn't really set the limits correctly
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    # Avoid overlapping title and axis labels
    plt.tight_layout()
    plt.savefig(saved_name)


def plot_scoring_func(y_test, score_t_test, thres_method, opt_thres, pred_labels,
                      plot_name, x_train, x_test, colorby_anom=True, start_stop=None, label="score",
                      score_tc_train=None, score_tc_test=None,
                      plot_channels=False):
    plot_channels = plot_channels & (score_tc_test is not None)
    if start_stop is None:
        start = 0
        stop = len(y_test)
        plot_name_full = plot_name + "_score_t.pdf"
    else:
        start = start_stop[0]
        stop = start_stop[1]
        plot_name_full = plot_name + "_" + str(start) + "_" + str(stop) + "_score_t.pdf"

    plt.close('all')
    cmap = plt.get_cmap('inferno')

    grid = 0
    if plot_channels:
        grid += 2*x_test.shape[1]  # data and reconstructions
    grid += 2  # true lables test, scores, and binary predictions
    fig, axes = plt.subplots(grid, 2, figsize=(15, 1 * grid))

    i = 0
    c0 = cmap(0)
    c1 = cmap(1 / 8)
    c2 = cmap(2/8)
    c3 = cmap(3/8)
    c4 = cmap(4/8)
    c5 = cmap(5/8)
    c6 = cmap(6/8)
    # axes[i].set_title('test data')
    if plot_channels:
        for channel_num, col in enumerate(x_test.T):
            j = i + 1
            axes[i, 0].set_title('train actual channel {}'.format(channel_num))
            axes[i, 0].plot(x_train[:, channel_num], color=c0)
            axes[i, 1].set_title('test actual channel {}'.format(channel_num))
            axes[j, 1].set_title('test {} channel {}'.format(label, channel_num))
            if colorby_anom:
                anom = np.ma.masked_where(y_test == 1, col)
                normal = np.ma.masked_where(y_test != 1, col)
                axes[i, 1].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
                preds = score_tc_test[:, channel_num]
                anom = np.ma.masked_where(y_test == 1, preds)
                normal = np.ma.masked_where(y_test != 1, preds)
                axes[j, 1].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
            else:
                axes[i, 1].plot(col[start:stop], color=c2)
                axes[j, 1].plot(score_tc_test[start:stop, channel_num], color=c3)
            if score_tc_train is not None:
                axes[j, 0].set_title('train {} channel {}'.format(label, channel_num))
                axes[j, 0].plot(score_tc_train[:, channel_num], color=c1)
            i += 2

    axes[i, 0].set_title('test true labels')
    axes[i, 0].plot(range(start, stop, 1), y_test[start:stop], color=c4)
    axes[i, 1].set_title('test true labels')
    axes[i, 1].plot(range(start, stop, 1), y_test[start:stop], color=c4)
    i += 1
    axes[i, 0].set_title('Anomaly score by {}'.format(label))
    axes[i, 1].set_title('binary labels for {} on {}'.format(thres_method, label))
    if colorby_anom:
        anom = np.ma.masked_where(y_test == 1, score_t_test)
        normal = np.ma.masked_where(y_test != 1, score_t_test)
        axes[i, 0].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
        anom = np.ma.masked_where(y_test == 1, pred_labels)
        normal = np.ma.masked_where(y_test != 1, pred_labels)
        axes[i, 1].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
    else:
        axes[i, 0].plot(range(start, stop, 1), score_t_test[start:stop], color=c5)
        axes[i, 1].plot(range(start, stop, 1), pred_labels[start:stop], color=c6)

    axes[i, 0].plot(range(start, stop, 1), [opt_thres] * int(stop - start), color=c0)
    plt.tight_layout()
    # fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.savefig(plot_name_full)


def plot_reconstruction_errors(x_train, x_test, y_test, train_raw_preds, test_raw_preds, score, name, folder,
                               colorby_anom=True, start_stop=None, label="error"):
    x_train = x_train.values
    if start_stop is None:
        start = 0
        stop = len(x_test)
    else:
        start = start_stop[0]
        stop = start_stop[1]

    plt.close('all')
    cmap = plt.get_cmap('inferno')

    grid = 0
    grid += 2*x_test.values.shape[1]  # data and reconstructions
    grid += 1  # true lables test
    grid += 3 * (len(score.keys()) // 2 + 1)
    fig, axes = plt.subplots(grid, 2, figsize=(15, 1 * grid))

    i = 0
    c0 = cmap(0)
    c1 = cmap(1 / 8)
    c2 = cmap(2/8)
    c3 = cmap(3/8)
    c4 = cmap(4/8)
    # axes[i].set_title('test data')
    for channel_num, col in enumerate(x_test.values.T):
        axes[i, 0].set_title('train actual channel {}'.format(channel_num))
        axes[i, 0].plot(x_train[:, channel_num], color=c0)
        axes[i, 1].set_title('test actual channel {}'.format(channel_num))
        if colorby_anom:
            anom = np.ma.masked_where(y_test == 1, col)
            normal = np.ma.masked_where(y_test != 1, col)
            axes[i, 1].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
        else:
            axes[i, 1].plot(col[start:stop], color=c2)
        i += 1
        axes[i, 0].set_title('train {} channel {}'.format(label, channel_num))
        axes[i, 0].plot(train_raw_preds[:, channel_num], color=c1)
        axes[i, 1].set_title('test {} channel {}'.format(label, channel_num))
        if colorby_anom:
            preds = test_raw_preds[:, channel_num]
            anom = np.ma.masked_where(y_test == 1, preds)
            normal = np.ma.masked_where(y_test != 1, preds)
            axes[i, 1].plot(range(start, stop, 1), anom[start:stop], range(start, stop, 1), normal[start:stop])
        else:
            axes[i, 1].plot(test_raw_preds[start:stop, channel_num], color=c3)
        i += 1

    axes[i, 0].set_title('test true labels')
    axes[i, 0].plot(y_test.values[start:stop], color=c4)
    axes[i, 1].set_title('test true labels')
    axes[i, 1].plot(y_test.values[start:stop], color=c4)
    i += 1
    for num, key in enumerate(score.keys()):
        col = num % 2
        c = cmap((5+num)/8)
        axes[i, col].set_title('raw scores for {}'.format(key))
        axes[i, col].plot(score[key][start:stop], color=c)
        i += 1
        for thres_method in ["top_k_time", "best_f1_test"]:
            opt_thres, pred_labels = threshold_and_predict(score[key], y_test, thres_method=thres_method)
            axes[i, col].set_title('binary labels for {} on {}'.format(thres_method, key))
            axes[i, col].plot(pred_labels[start:stop], color=c)
            i += 1
        if col == 0:
            i -= 3

    plt.tight_layout()
    # fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.savefig(os.path.join(folder, f"{label}_{name}.pdf"))


def plot_train_val_loss(additional_params_filename, folder_name, name= ""):
    with open(additional_params_filename, "rb") as file:
        additional_params = pickle.load(file)
    train_loss = additional_params["train_loss_per_epoch"]
    val_loss = additional_params["val_loss_per_epoch"]
    fig = plt.figure()
    x = range(len(train_loss))
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, val_loss, label="val_loss")
    plt.legend()
    fig.savefig(os.path.join(folder_name, "train_val_loss_" + name + ".pdf"))


def plot_roc(y_test, test_scores, name, save_dir):
    # plot and save the roc curve
    fpr, tpr, thres = roc_curve(y_test, test_scores, pos_label=1)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(fpr) <= 20000:
        ax.plot(fpr, tpr, 'o', label="ROC-curve")
    else:
        ax.plot(fpr, tpr, label="ROC-curve")
    ax.set_title(name)
    # todo: the following doesn't really set the limits correctly
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()
    # Avoid overlapping title and axis labels
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ROC curve of {name}.pdf"))


def get_prc_roc_aucs(y_test, test_scores):
    roc_auc = roc_auc_score(y_test, test_scores)
    prec, rec, _ = precision_recall_curve(y_test, test_scores, pos_label=1)
    prc_auc = auc(rec, prec)
    return prc_auc, roc_auc


def get_npt_labels(score_t_test, y_test, config):
    nptconfig = NptConfig(config)
    channel = Channel(nptconfig)
    channel.y_test = np.zeros(len(y_test))
    channel.y_hat = score_t_test
    errors = Errors(channel, nptconfig, run_id="", apply_shift=False, ignore_y_test=True)
    errors.process_batches(channel)
    pred_ano_seq = errors.E_seq
    pred_labels = np.zeros(len(y_test))
    for start, end in pred_ano_seq:
        pred_labels[start:(end + 1)] = 1
    return pred_labels


class NptConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def get_version_order(version):
    if version == "Error":
        return 0
    elif version == "Gauss-S":
        return 1
    elif version == "Gauss-D":
        return 2
    elif version == "Gauss-D-K":
        return 3
    else:
        return -1


def order_grouped_results(grouped_resuts_mean):
    grouped_resuts_mean["base_algo"] = grouped_resuts_mean["algo"].apply(
        lambda algo: algo.replace("-dyn-gauss-conv", "").replace("-R", "").replace("-dyn", "").replace("-Gauss-S", ""))
    grouped_resuts_mean["version"] = grouped_resuts_mean["algo"].apply(lambda algo_name: get_algo_version(algo_name))
    grouped_resuts_mean["version_order"] = grouped_resuts_mean["version"].apply(
        lambda version: get_version_order(version))
    grouped_resuts_mean = grouped_resuts_mean.sort_values(by="version_order")
    return grouped_resuts_mean


def get_algo_version(algo_name):
    if algo_name.endswith("-R"):
        version = "Error"
    elif algo_name.endswith("-dyn-gauss-conv"):
        version = "Gauss-D-K"
    elif algo_name.endswith("-dyn"):
        version = "Gauss-D"
    elif algo_name.endswith("-Gauss-S"):
        version = "Gauss-S"
    elif algo_name in ['AutoEncoder_recon_all', 'LSTM-ED_recon_all', 'PcaRecons',
       'RawSignalBaseline', 'UnivarAutoEncoder_recon_all', 'VAE_LSTM', 'TcnED']:
        version = "Gauss-S"
    else:
        version = "base"
    return version


def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

