import copy

best_configs = {}
constant_seed = 42
num_epochs = 100
batch_size = 128
patience = 10
lr = 1e-3
gpu = 0
train_val_pc = 0.25
pca_expl_var = 0.9
n_dims = {"swat-long": 51, "swat": 51, "wadi": 123, "damadics": 32, "damadics-s": 32, "msl": 55, "smap": 25, "smd": 38,
          "skab": 8}
strides = {"swat-long": 10, "swat": 10, "wadi": 10, "damadics": 10, "damadics-s": 10, "msl": 1, "smap": 1, "smd": 1,
           "skab": 1}
seq_lens = {"swat-long": 100, "swat": 100, "wadi": 30, "damadics": 100, "damadics-s": 100, "msl": 100, "smap": 100,
            "smd": 100, "skab": 100}
datasets_config = {"swat-long": {"shorten_long": False}, "damadics-s": {"drop_init_test": True}}
thres_methods = ["top_k_time",
                 "best_f1_test",
                 "tail_prob_1",
                 "tail_prob_2",
                 "tail_prob_3",
                 "tail_prob_4",
                 "tail_prob_5"
                 ]

seed = 42
constant_std = 0.000001

# thresholding
long_windows = {"swat-long": 100000, "swat": 100000, "wadi": 100000, "damadics": 100000, "msl": 2000, "smap": 2000,
                "smd": 25000, "damadics-s": 100000, "skab": 100}
kernel_sigmas = {"swat-long": 120, "swat": 120, "wadi": 120, "damadics": 5, "msl": 10, "smap": 10,
                "smd": 1, "damadics-s": 5, "skab": 1}

default_thres_config = {"top_k_time": {},
                        "best_f1_test": {"exact_pt_adj": True},
                        "thresholded_score": {},
                        "tail_prob": {"tail_prob": 2},
                        "tail_prob_1": {"tail_prob": 1},
                        "tail_prob_2": {"tail_prob": 2},
                        "tail_prob_3": {"tail_prob": 3},
                        "tail_prob_4": {"tail_prob": 4},
                        "tail_prob_5": {"tail_prob": 5},
                        "dyn_gauss": {"long_window": 10000, "short_window": 1, "kernel_sigma": 10},
                        "nasa_npt": {"batch_size": 70, "window_size": 30, "telem_only": True,
                                     "smoothing_perc": 0.005, "l_s": 250, "error_buffer":5, "p": 0.05}}


def get_thres_config(ds_name):
    thres_config = copy.deepcopy(default_thres_config)
    thres_config["dyn_gauss"]["short_window"] = 1
    thres_config["dyn_gauss"]["long_window"] = long_windows[ds_name]
    thres_config["dyn_gauss"]["kernel_sigma"] = kernel_sigmas[ds_name]
    for i in range(1,6):
        if ds_name not in ["msl", "smap"]:
            thres_config["tail_prob_" + str(i)]["tail_prob"] = i*n_dims[ds_name]
        else:
            thres_config["tail_prob_" + str(i)]["tail_prob"] = i
    thres_config["nasa_npt"]["l_s"] = seq_lens[ds_name]
    return thres_config


def get_best_config(algo_name, ds_name):
    best_configs["RawSignalBaseline"] = {}
    best_configs["PcaRecons"] = {"explained_var": pca_expl_var}
    best_configs["AutoEncoder_recon_all"] = {
        "sequence_length": seq_lens[ds_name],
        "num_epochs": num_epochs,
        "hidden_size": n_dims[ds_name]//2,
        "lr": 0.0001,
        "gpu": gpu,
        "batch_size": batch_size,
        "stride": strides[ds_name],
        "patience": patience,
        "train_val_percentage": train_val_pc,
        "last_t_only": False
    }

    best_configs["UnivarAutoEncoder_recon_all"] = {
        "sequence_length": seq_lens[ds_name],
         "num_epochs": num_epochs,
         "hidden_size": 5,
         "lr": lr,
         "gpu": gpu,
         "batch_size": 256,
         "stride": strides[ds_name],
         "patience": patience,
         "train_val_percentage": train_val_pc,
         "last_t_only": False,
         "n_processes": 3
    }

    best_configs["LSTM-ED_recon_all"] = {
        "num_epochs": num_epochs,
        "batch_size": 64,
        "lr": lr,
        "hidden_size": None,
        "sequence_length": seq_lens[ds_name],
        "train_val_percentage": train_val_pc,
        "n_layers": (1, 1),
        "gpu": gpu,
        "stride": strides[ds_name],
        "patience": patience,
        "explained_var": pca_expl_var,
        "set_hid_eq_pca": True,
        "last_t_only": False,
    }

    # to test and confirm if this is ok for swat, not yet tested
    best_configs["TelemanomAlgo"] = {
        "sequence_length": seq_lens[ds_name],
        "num_epochs": num_epochs,
        "gpu": gpu,
        "batch_size": 64,
        "stride": strides[ds_name],
        "train_val_percentage": train_val_pc,
        "test_batch_size": 256 if ds_name == "wadi" else 512,
        "n_lstm_units": n_dims[ds_name],
        "telem_only": True if ds_name in ["msl", "smap"] else False,
        "verbose": True
    }

    best_configs["TcnED"] = {
        "num_epochs": num_epochs,
        # val_err 0.03
        "lr": 0.00015,
        "dropout": 0.42,
        "num_channels": [min([n_dims[ds_name]//6, 10])]*3,
        "batch_size": 128,
        "sequence_length": seq_lens[ds_name],
        "stride": strides[ds_name],
        "gpu": gpu,
        "train_val_percentage": train_val_pc,
        "patience": patience
    }

    best_configs["VAE-LSTM"] = {
        "num_epochs": num_epochs,
        "lr": 0.0095,
        "batch_size": batch_size,
        "sequence_length": seq_lens[ds_name],
        "stride": strides[ds_name],
        "gpu": gpu,
        "train_val_percentage": train_val_pc,
        "patience": patience,
        "intermediate_dim": 15,
        "z_dim": 3,
        "n_dim": n_dims[ds_name],
        "REG_LAMBDA": 0.55,
        "kulback_coef": 0.28
    }

    best_configs["MSCRED"] = {
        "sequence_length": seq_lens[ds_name],
        "num_epochs": num_epochs,
        "lr": 0.0001,
        "gpu": gpu,
        "batch_size": batch_size,
        "stride": strides[ds_name],
        "explained_var": pca_expl_var if ds_name=='wadi' else None
    }

    best_configs["BeatGan"] = {
        "sequence_length": seq_lens[ds_name],
        "num_epochs": num_epochs,
        "hidden_size": 10,
        "lr": 0.0001,
        "gpu": gpu,
        "batch_size": batch_size,
        "stride": strides[ds_name],
        "beta1": 0.5
    }

    best_configs["OCAN_direct_seq"] = {
        "lr": 0.0001,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "direct_seq": True,
        "encode_seq": False,
        "sequence_length": 10,
        "hidden_size": None,
        "gpu": None,
        "stride": strides[ds_name],
        "patience": None,
        "train_val_percentage": None
    }

    best_configs["OCAN_encode_seq"] = {
        "lr": 0.0001,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "direct_seq": False,
        "encode_seq": True,
        "sequence_length": seq_lens[ds_name],
        "hidden_size": 200,
        "gpu": gpu,
        "stride": strides[ds_name],
        "patience": patience,
        "train_val_percentage": train_val_pc
    }

    best_configs["OCAN"] = {
        "lr": 0.0001,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "encode_seq": False,
        "direct_seq": False
    }

    best_configs["OmniAnoAlgo"] = {
        "sequence_length": seq_lens[ds_name],
        "num_epochs": 20,
        "stride": strides[ds_name],
        "z_dim": 3,
        "batch_size": 50,
        "rnn_num_hidden": 500,
        "dense_dim": 500,
        "nf_layers": 20,
        "train_val_percentage": train_val_pc
    }    

    best_configs["Ocsvm_gpu"] = {
        "explained_var": pca_expl_var,
        "univar": False,
        "sequence_length": seq_lens[ds_name],
        "stride": strides[ds_name],
        "train_val_percentage": train_val_pc,
        "batch_size": batch_size, 
        "gamma": "auto",
        "nu": 0.48899475599830133
    }      

    return best_configs[algo_name]

