from sklearn.metrics import classification_report
from my_data_functions import load_data_partial, get_results, load_edBB_all
from src.algorithms import AutoEncoder
from src.algorithms.algorithm_utils import get_sub_seqs
from my_experiments_mvts import get_normalized_scores, get_fitted_scores, collect_results, setup_out_dir 
import numpy as np
import pandas as pd

# from src.algorithms.algorithm_utils import fit_univar_distr
# from src.evaluation.evaluation_utils import get_scores_channelwise, threshold_and_predict
# from datetime import datetime
# import os
# from sklearn.metrics import recall_score, classification_report
# import logging
# from src.datasets.dataset import get_events

# Global configs
body_parts = ['full', 'upper']
body_part = body_parts[1]
sequence_length = 15
num_epochs = 50
hidden_size = 10
n_layers_ed = (10,10)
batch_size = 16
learning_rate = 0.0001
seed = 0
n_data_folds = 3

def get_model(model_name,features_dim, out_dir=None):

    if model_name == 'AutoEncoder':
        model = AutoEncoder(sequence_length=sequence_length, num_epochs=num_epochs, hidden_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'UnivarAutoEncoder':
    #     model = UnivarAutoEncoder(sequence_length=sequence_length, num_epochs=num_epochs, hidden_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # # elif model_name == 'LSTMED':
    #     model = LSTMED(sequence_length=sequence_length,hidden_size=hidden_size,num_epochs=num_epochs,batch_size=batch_size,lr=learning_rate,n_layers=n_layers_ed,seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'VAE_LSTM':
    #     model = VAE_LSTM(sequence_length=sequence_length, num_epochs= num_epochs,n_dim=features_dim, intermediate_dim=2*hidden_size, z_dim=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'TcnED':
    #     model = TcnED(sequence_length=sequence_length,num_epochs= num_epochs, num_channels=[features_dim],kernel_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'MSCRED':
    #     model = MSCRED(sequence_length=sequence_length, num_epochs=num_epochs, lr=learning_rate, batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'OmniAnoAlgo':
    #     model = OmniAnoAlgo(sequence_length=sequence_length, num_epochs=num_epochs,z_dim=hidden_size, batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'PcaRecons':
    #     model = PcaRecons(seed=seed, out_dir=out_dir)    
    # elif model_name == 'RawSignalBaseline':
    #     model = RawSignalBaseline(seed=seed, out_dir=out_dir)
    return model

def apply_threshold(train_scores, test_scores):
    thresh = np.mean(train_scores) + np.std(train_scores)
    train_labels = np.where(train_scores < thresh, 0, 1)
    test_labels = np.where(test_scores < thresh, 0, 1)
    return thresh, train_labels, test_labels

def partition_label_indecies(labels, seq_len):
    zero_idxs = set([])
    one_idxs = set([])
    i = 0 
    while i <= len(labels)-seq_len:
        if sum(labels[i:i+seq_len]) > 0:
            one_idxs.add(i)
        else:
            zero_idxs.add(i)

        i += 1
    return np.array(sorted(zero_idxs)), np.array(sorted(one_idxs))
# partition_label_indecies([0,0,0,1,0,1,0,0,0],3)

def experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, 
                        score_distr_name='', thres_method='top_k_time', algorithm='default'):
    print(f'\n\nprocessing folder {folder_idx}...')

    x_data, y_data = load_data_partial(dataset_name, folder_idx, feature_type, body_part, train_ratio=0.0)

    features_dim = x_data.shape[1]
    out_dir=setup_out_dir(dataset_name, model_name, feature_type, folder_idx)
    model = get_model(model_name,features_dim, out_dir=out_dir)

    x_seqs = get_sub_seqs(x_data.values, seq_len=sequence_length)
    y_seqs = np.array([1 if sum(y_data.iloc[i:i + sequence_length])>0 else 0 for i in range(len(x_seqs))])
    n_eter = 4
    train_ratio = 0.3
    n_train = int(len(x_seqs) * train_ratio)
    x_train_all = x_seqs[:n_train]
    y_train_all = y_seqs[:n_train]
    x_test_all = x_seqs[n_train:]
    y_test_all = y_seqs[n_train:]

    top_k = np.sum(y_seqs)
    i=0
    while  True:
        n_train = int(len(x_seqs) * train_ratio)
        x_train = x_seqs[:n_train]
        y_train = y_seqs[:n_train]
        x_test = x_seqs[n_train:]
        y_test = y_seqs[n_train:]

        if len(x_test) < top_k:
            break
            

        model.fit_sequences(x_train)
        test_preds = model.predict_sequences(x_test)
        train_preds = model.predict_sequences(x_train)
        if score_distr_name == 'normalized_error':
            test_scores = get_normalized_scores(train_preds['error_tc'], test_preds['error_tc'])
        else:
            if test_preds['score_t'] is None:
                train_scores, test_scores = get_fitted_scores(train_preds['error_tc'], test_preds['error_tc'])  
            else:
                train_scores, test_scores = test_preds['score_t'], train_preds['score_t']
        test_scores = test_scores[sequence_length-1:]
        print(f'results of {i+1}st iteration:')
        print('number of test sequences: ', len(x_test))
        # if i==0:
        aps, auroc, acc = get_results(y_test, test_scores, top_k= top_k)
        
        if algorithm == 'default':
            break

        test_idx = np.argsort(test_scores)
        x_seqs = x_test[test_idx]
        y_seqs = y_test[test_idx]
        i += 1
    
    
    return aps, auroc, acc

def experiments_on_dataset(dataset_name, model_name, feature_type, distr_name='normalized_error', algorithm='default', 
                            edBB_pretrain=False, edBB_finetune=False):
    pretrained_model = None
    if edBB_pretrain:
        print('training on edBB...')
        x_train, _ = load_edBB_all(feature_type, body_part)
        features_dim = x_train.shape[1]
        pretrained_model = get_model(model_name, features_dim, out_dir=setup_out_dir('edBB',model_name, feature_type))
        pretrained_model.fit(x_train)

    n=38
    if dataset_name == 'MyDataset':
        n=12
    aps_avg=[]
    auroc_avg = []
    acc_avg = []
    for folder_idx in range (1, n+1):
        aps,auroc, acc = experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, distr_name, algorithm=algorithm)
        aps_avg.append(aps)
        auroc_avg.append(auroc)
        acc_avg.append(acc)

    aps_avg = np.mean(aps_avg)
    auroc_avg = np.mean(auroc_avg)
    acc_avg = np.mean(acc_avg)

    print(f'\nAverage results on {dataset_name} of {model_name} by {feature_type} features:')
    print(f"APS: {aps_avg:0.3f}, AUROC: {auroc_avg:0.3f}, Precision: {acc_avg:0.3f}\n")
    return aps_avg, auroc_avg, acc_avg

if __name__ == '__main__':
    datasets = ['edBB', 'MyDataset']
    feature_types = ['original', 'angle', 'distance', 'angle_distance', 'angle_plus_distance']
    model_names = ['AutoEncoder', 'LSTMED', 'VAE_LSTM','UnivarAutoEncoder',  'OmniAnoAlgo','MSCRED', 'TcnED', 'PcaRecons', 'RawSignalBaseline']
    distr_names = ['normalized_error', 'univar_gaussian']#, 'univar_lognormal', 'univar_lognorm_add1_loc0', 'chi']
    thresh_methods = ['top_k_time']#, 'best_f1_test', 'tail_prob']
    dataset_name, model_name, folder_idx, feature_type = datasets[1], model_names[0], 1, feature_types[2]
    experiment_on_folder(dataset_name, model_name, folder_idx, feature_type=feature_type, score_distr_name=distr_names[0])
    # experiments_on_dataset(dataset_name, model_name, feature_type, distr_names[1], algorithm='default')
    # experiments_on_dataset(dataset_name, model_name, feature_type, distr_names[1], algorithm='algo_2')
