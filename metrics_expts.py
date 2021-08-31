from src.datasets import MultiEntityDataset

from configs import datasets_config, get_thres_config
from src.evaluation.evaluator import *
from src.evaluation.evaluation_utils import *
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def run_random_scoring(root_dir, ds=["damadics-s", "msl", "smap", "smd", "swat", "wadi"]):
    os.makedirs(root_dir, exist_ok=True)
    init_logging(os.path.join(root_dir, 'logs'))
    logger = logging.getLogger(__name__)

    seed = 0
    np.random.seed(seed)
    fscores_dict_raw_t = dict()
    fscores_dict_avg_t = dict()
    fscores_dict_raw_c = dict()
    fscores_dict_avg_c = dict()

    fscores_dict_raw_pa = dict()
    fscores_dict_avg_pa = dict()

    for ds_name in ds:
        if ds_name in datasets_config.keys():
            ds_kwargs = datasets_config[ds_name]
        else:
            ds_kwargs = {}
        print(ds_name)
        ds_class = get_dataset_class(ds_name)
        ds_multi = MultiEntityDataset(dataset_class=ds_class, seed=seed, ds_kwargs=ds_kwargs)
        thres_config_dict = get_thres_config(ds_name)
        fscores_pa = []
        fscores_t = []
        fscores_c = []
        for entity in ds_multi.datasets:
            _, _, _, y_test = entity.data()

            true_events = get_events(y_test)
            # random scores
            score_t_test = np.random.rand(len(y_test))
            thres_method = "best_f1_test"
            test_anom_frac_entity = entity.get_anom_frac_entity()
            _, pred_labels = threshold_and_predict(score_t_test, y_test, true_events=true_events, logger=logger,
                                                           thres_method=thres_method, point_adjust=True,
                                                           score_t_train=None, thres_config_dict=thres_config_dict,
                                                   test_anom_frac=test_anom_frac_entity)
            print("num predicted by pa {}".format(sum(pred_labels)/len(pred_labels)))
            _, _, _, _, _, fscore_pa = evaluate_predicted_labels(pred_labels, y_test, logger=logger,
                                                                      true_events=true_events,
                                                                      eval_method="time-wise", point_adjust=True)

            _, pred_labels = threshold_and_predict(score_t_test, y_test, true_events=true_events, logger=logger,
                                                           thres_method=thres_method, point_adjust=False,
                                                           score_t_train=None, thres_config_dict=thres_config_dict,
                                                   test_anom_frac=test_anom_frac_entity)
            print("num predicted by time-wise f1 {}".format(sum(pred_labels)/len(pred_labels)))
            _, _, _, _, _, fscore_t = evaluate_predicted_labels(pred_labels, y_test, logger=logger,
                                                                      true_events=true_events,
                                                                      eval_method="time-wise", point_adjust=False)

            _, pred_labels = threshold_and_predict(score_t_test, y_test, true_events=true_events,
                                                                            logger=logger,
                                                           thres_method=thres_method, point_adjust=False,
                                                           score_t_train=None, thres_config_dict=thres_config_dict,
                                                           return_auc=False, composite_best_f1=True,
                                                   test_anom_frac=test_anom_frac_entity)
            print("num predicted by fc_1 {}".format(sum(pred_labels)/len(pred_labels)))
            fscore_c = get_composite_fscore_raw(pred_labels, true_events, y_test)
            fscores_pa.append(fscore_pa)
            fscores_t.append(fscore_t)
            fscores_c.append(fscore_c)
            print(fscore_pa)
            print(fscore_t)
            print(fscore_c)
        fscores_dict_raw_pa[ds_name] = fscores_pa
        fscores_dict_avg_pa[ds_name] = np.nanmean(fscores_pa)
        fscores_dict_raw_t[ds_name] = fscores_t
        fscores_dict_avg_t[ds_name] = np.nanmean(fscores_t)
        fscores_dict_raw_c[ds_name] = fscores_c
        fscores_dict_avg_c[ds_name] = np.nanmean(fscores_c)

    print(fscores_dict_avg_pa)
    print(fscores_dict_avg_t)
    print(fscores_dict_avg_c)
    df = pd.DataFrame([fscores_dict_avg_t, fscores_dict_avg_pa, fscores_dict_avg_c])
    # df["Point adjust"] = [False, True, False]
    df["Metric"] = ["Time-wise F1", "Point-Adjust F1", "Composite F1"]
    df.set_index("Metric", inplace=True)
    print(df)
    print(df.to_latex())
    df.to_csv(os.path.join(root_dir, "pt_adj_composite_2ds.csv"))


def run_examples_scores(root_dir):
    init_logging(os.path.join(root_dir, 'logs'))
    logger = logging.getLogger(__name__)
    seed = 0
    np.random.seed(seed)
    fscores_t = []
    fscores_fc = []
    fscores_pa = []
    len_ts = 150
    scores_zero = np.zeros(len_ts)
    scores_ones = scores_zero + 1
    scores_rand = np.random.choice([0, 1], len_ts)
    y_test = np.zeros(len_ts)
    y_test[10:30] = 1
    y_test[60:70] = 1
    scores_1 = np.concatenate([[0, 0, 0, 0, 1] * 20])
    scores_2 = np.zeros(len_ts)
    scores_2[[11, 12, 50, 65, 68]] = 1
    scores_3 = np.zeros(len_ts)
    scores_3[15] = 1
    scores_3[65] = 1
    scores_3[40:50] = 1
    scores_3[110:120] = 1
    scores_4 = np.zeros(len_ts)
    scores_4[12:18] = 1
    scores_4[62:70] = 1
    scores_5 = np.zeros(len_ts)
    scores_5[12:28] = 1
    scores = np.stack([scores_rand, scores_ones, scores_3, scores_5, scores_2, scores_4])
    true_events = get_events(y_test)
    labels = ["Random", "All positives", "$rec_e$=1, high $FP_t$", "$rec_e$=0.5, no $FP_t$", "$rec_e$=1, 1 $FP_t$", "$rec_e$=1, no $FP_t$"]
    for pred_num in range(scores.shape[0]):
        pred_labels = scores[pred_num, :]
        _, _, _, _, _, fscore_pa = evaluate_predicted_labels(pred_labels, y_test, logger=logger,
                                                             true_events=true_events,
                                                             eval_method="time-wise", point_adjust=True)

        _, _, _, _, _, fscore_t = evaluate_predicted_labels(pred_labels, y_test, logger=logger,
                                                            true_events=true_events,
                                                            eval_method="time-wise", point_adjust=False)

        fscore_c = get_composite_fscore_raw(pred_labels, true_events, y_test)

        fscores_t.append(fscore_t)
        fscores_fc.append(fscore_c)
        fscores_pa.append(fscore_pa)
    print(fscores_t)
    print(fscores_fc)
    print(fscores_pa)
    df = pd.DataFrame(data=np.stack([fscores_t, fscores_pa, fscores_fc],axis=-1),
                      columns=["Time-wise F1", "Point-Adjust F1", "Composite F1"])
    print(df)
    fig, axes = plt.subplots(7, figsize=(4, 3))
    normal = np.ma.masked_where(y_test != 1, y_test)
    axes[0].plot(range(len_ts), y_test, range(len_ts), normal, linewidth=1)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_ylabel("$y$: Ground truth", rotation=0, labelpad=45, fontsize='large')
    axes[0].set_xlim(0, 150)
    for pred_num in range(scores.shape[0]):
        preds = scores[pred_num, :]
        axes[pred_num+1].plot(range(len_ts), preds, linewidth=1)
        tp = np.where((y_test == 1) & (preds == 1))[0]
        axes[pred_num+1].scatter(tp, np.ones(len(tp)), color='brown', s=5)
        axes[pred_num + 1].set_ylim(-0.1, 1.1)
        axes[pred_num + 1].set_xlim(0, 150)
        axes[pred_num + 1].set_xticklabels([])
        axes[pred_num + 1].set_yticklabels([])
        axes[pred_num + 1].set_ylabel("$\haty_{}$: {}".format(pred_num+1, labels[pred_num]), rotation=0, labelpad=45,
                                      fontsize='large')
    df = df.round(4)
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.tight_layout()
    print(df.to_latex())
    df.to_csv(os.path.join(root_dir, "metrics_eval_new.csv"))
    plt.savefig(os.path.join(root_dir, "example_scores.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    print(os.getcwd())
    ds = ["msl", "smap", "smd", "damadics-s"]
    root_dir = os.path.join(os.getcwd(), "reports", "metrics_expts")
    os.makedirs(root_dir, exist_ok=True)
    run_random_scoring(root_dir, ds=ds)
    run_examples_scores(root_dir)





