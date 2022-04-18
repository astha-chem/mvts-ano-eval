import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import glob

# update: 2022-03-30

def load_edBB_all(feature_type, body_part, normals_only=True):
    n = 38
    x_all, y_all = [], []
    for folder_idx in range(1, n+1):
        x_folder, y_folder = load_data_partial('edBB', folder_idx, feature_type, body_part, train_ratio=1.0)
        x_all.append(x_folder.to_numpy())
        y_folder = y_folder.to_numpy().reshape((-1,1))
        y_all.append(y_folder)
    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    if normals_only:
        idxs = np.where(y_all == 0)[0]
        x_all = x_all[idxs]
        y_all = y_all[idxs]
    return pd.DataFrame(x_all), pd.DataFrame(y_all)

def load_data_partial(dataset, folder_idx, feature_type, body_part, train_ratio=0.3, return_filenames=False):

    base_path = 'E:/Atabay/Datasets/edBB/Side/_data'
    if dataset == 'MyDataset':
        base_path = f'E:/Atabay/Datasets/MyDataset/{folder_idx:02d}_*'
        base_path = glob.glob(base_path)[0]

    if feature_type == 'original':
        #load skeletons
        coordinates_path = base_path + f'/coordinates_movnet/{folder_idx:02d}.csv'
        if dataset == 'MyDataset':
            coordinates_path = base_path + '/coordinates_movnet/01.csv'
        time_series = pd.read_csv(coordinates_path, header=None)
        time_series = time_series.iloc[:,1:]
        if body_part == 'upper':
            time_series = time_series.iloc[:,:22]
    else:
        features_path = base_path + f'/angle_distance_features/{folder_idx:02d}.csv'
        if dataset == 'MyDataset':
            features_path = base_path + f'/features/angle_distance_features.csv'
        data = pd.read_csv(features_path, header=None)
        data = data.iloc[:,1:]
        if feature_type == 'angle_distance':
            time_series = data
            if body_part == 'upper':
                time_series = time_series.iloc[:, :22]
        elif feature_type == 'angle_plus_distance':
            data = data.to_numpy()
            ts1 = data[:,list(range(0, data.shape[1], 2))]
            ts2 = data[:,list(range(1, data.shape[1], 2))]
            time_series =pd.DataFrame(ts1+ts2)
            if body_part == 'upper':
                time_series = time_series.iloc[:, :11]
        else:
            if feature_type == 'angle':
                rem = 0
            else:
                rem = 1
            time_series = data.iloc[:,list(range(rem, data.shape[1], 2))]
            if body_part == 'upper':
                time_series = time_series.iloc[:, :11]
    
    #load labels
    labels_path=base_path+f'/labels_movnet/{folder_idx:02d}.csv'
    if dataset == 'MyDataset':
        labels_path = base_path + '/labels.csv'

    labels = pd.read_csv(labels_path, header=None)
    if return_filenames:
        filenames = labels.iloc[:,0]
    labels = labels.iloc[:,1]

    if train_ratio == 1.0 or train_ratio == 0.0:
        return time_series, labels

    n_data = len(labels)
    n_train = int(n_data * train_ratio)
    
    x_train = time_series.iloc[:n_train]    
    y_train = labels.iloc[:n_train]
    x_test = time_series.iloc[n_train:]
    y_test = labels.iloc[n_train:]
    if return_filenames:
        test_filenames = filenames.iloc[n_train:]

    if y_train.sum() > 0:
        idxs = np.where(y_train.to_numpy() == 1)[0]
        idxs2 = np.where(y_test.to_numpy() == 0)[0][:len(idxs)]
        x_train2 = x_train.to_numpy()
        x_test2 = x_test.to_numpy()
        x_train2[idxs] = x_test2[idxs2]
        x_test2[idxs2] = x_train.iloc[idxs].values
        x_train = pd.DataFrame(x_train2)
        x_test = pd.DataFrame(x_test2)
        # xs = x_train.iloc[idxs,:]
        
        # x_train.iloc[idxs,:] = x_test.iloc[idxs2,:].values
        # x_test.iloc[idxs2,:] = xs.values
        y_train.iloc[idxs] = 0
        y_test.iloc[idxs2] = 1
        if return_filenames:
            test_filenames.iloc[idxs2] = filenames.iloc[idxs]
            return x_train, y_train, x_test, y_test, test_filenames

        print('some records of training data exchanged.')

    return x_train, y_train, x_test, y_test

def get_results(y_real, pred_scores, top_k, print_results=True):
    aps = average_precision_score(y_real, pred_scores, pos_label=1)
    roc = roc_auc_score(y_real, pred_scores)
    idxs = np.argsort(pred_scores)
    idxs = idxs[-top_k:]
    y_pred = y_real[idxs]
    acc = np.sum(y_pred) / top_k
    if print_results:
        print(f"APS: {aps:0.3f}, AUROC: {roc:0.3f}, Top-{top_k} Accuracy: {acc:0.3f}")
    return aps, roc, acc

def test(dataset, feature_type):
    x_train, y_train, x_test, y_test = load_data_partial(dataset,folder_idx=1,feature_type=feature_type,body_part="upper")
    print('loaded from',dataset,', features:',feature_type)        
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')

if __name__ == '__main__':
    # test('MyDataset','distance')
    x, y = load_edBB_all('original','upper')   
    print('X:',x.shape,'y:',y.shape)
 