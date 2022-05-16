# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 09:09:09 2022

@author: CS-GKU
"""

import numpy as np
import glob
import pandas as pd
import os

Nose, Nose_idx = [0, 1], 0
LEye, LEye_idx = [2, 3], 1
REye, REye_idx = [4, 5], 2
LEar, LEar_idx = [6, 7], 3
REar, REar_idx = [8, 9], 4
LShoulder, LShoulder_idx = [10, 11], 5
RShoulder, RShoulder_idx = [12, 13], 6
LElbow, LElbow_idx = [14, 15], 7
RElbow, RElbow_idx = [16, 17], 8
LWrist, LWrist_idx = [18, 19], 9
RWrist, RWrist_idx = [20, 21], 10
LHip, LHip_idx = [22, 23], 11
RHip, RHip_idx = [24, 25], 12
LKnee, LKnee_idx = [26, 27], 13
RKnee, RKnee_idx = [28, 29], 14
LAnkle, LAnkle_idx = [30, 31], 15
RAnkle, RAnkle_idx = [32, 33], 16



def get_angle_distance(p0, p1, origin_angle, feature_type='angle_dist'):
    dx = p1[0]-p0[0]
    dy = p1[1]-p0[1]
    distance = ((dx)**2 + (dy)**2)**0.5
    angle = np.arctan2(dx,dy) - origin_angle
    if feature_type == 'array':
        return dx, dy        
    return angle, distance

def get_features(coordinates, feature_type):
    features = []
    pOrigin = (coordinates[LShoulder] + coordinates[RShoulder])/2
    aOrigin = 0
    p1 = coordinates[Nose]
    features.append(get_angle_distance(pOrigin, p1, aOrigin, feature_type))
    
    p0 = p1
    p1 = coordinates[LEye]
    features.append(get_angle_distance(p0, p1, features[Nose_idx][0], feature_type))
    p1 = coordinates[REye]
    features.append(get_angle_distance(p0, p1, features[Nose_idx][0], feature_type))
    
    p0 = coordinates[LEye]
    p1 = coordinates[LEar]
    features.append(get_angle_distance(p0, p1, features[LEye_idx][0], feature_type))
    p0 = coordinates[REye]
    p1 = coordinates[REar]
    features.append(get_angle_distance(p0, p1, features[REye_idx][0], feature_type))

    p1 = coordinates[LShoulder]
    features.append(get_angle_distance(pOrigin, p1, aOrigin, feature_type))
    p1 = coordinates[RShoulder]
    features.append(get_angle_distance(pOrigin, p1, aOrigin, feature_type))
    p0 = coordinates[LShoulder]
    p1 = coordinates[LElbow]
    features.append(get_angle_distance(p0, p1, features[LShoulder_idx][0], feature_type))
    p0 = coordinates[RShoulder]
    p1 = coordinates[RElbow]
    features.append(get_angle_distance(p0, p1, features[RShoulder_idx][0], feature_type))
    p0 = coordinates[LElbow]
    p1 = coordinates[LWrist]
    features.append(get_angle_distance(p0, p1, features[LElbow_idx][0], feature_type))
    p0 = coordinates[RElbow]
    p1 = coordinates[RWrist]
    features.append(get_angle_distance(p0, p1, features[RElbow_idx][0], feature_type))

    p0 = coordinates[LShoulder]
    p1 = coordinates[LHip]
    features.append(get_angle_distance(p0, p1, features[LShoulder_idx][0], feature_type))
    p0 = coordinates[RShoulder]
    p1 = coordinates[RHip]
    features.append(get_angle_distance(p0, p1, features[RShoulder_idx][0], feature_type))
    p0 = coordinates[LHip]
    p1 = coordinates[LKnee]
    features.append(get_angle_distance(p0, p1, features[LHip_idx][0], feature_type))
    p0 = coordinates[RHip]
    p1 = coordinates[RKnee]
    features.append(get_angle_distance(p0, p1, features[RHip_idx][0], feature_type))
    p0 = coordinates[LKnee]
    p1 = coordinates[LAnkle]
    features.append(get_angle_distance(p0, p1, features[LKnee_idx][0], feature_type))
    p0 = coordinates[RKnee]
    p1 = coordinates[RAnkle]
    features.append(get_angle_distance(p0, p1, features[RKnee_idx][0], feature_type))

    return features


feature_types = ['angle_dist','array']
type_idx = 1
datasets_name = ['edBB', 'MyDataset']
dataset_idx = 0
# base_path = f'E:/Atabay/Datasets/{datasets_name[dataset_idx]}/'
base_path = f'data/{datasets_name[dataset_idx]}/'
if datasets_name[dataset_idx] == 'edBB':
    # base_path += 'Side/_data/'
    for i in range(1, 39):
        print(f'processing folder {i}')
        coordinates = pd.read_csv(base_path + f'coordinates_movnet/{i:02d}.csv', header=None)
        features = coordinates.copy()
        n = len(coordinates)
        for j in range(n):
            feat = np.array(get_features(coordinates.iloc[j,1:].to_numpy(), feature_types[type_idx]))
            features.iloc[j,1:] = feat.reshape((-1,))
        if feature_types[type_idx] == 'array':
            features.to_csv(base_path+f'array_features/{i:02d}.csv', header=None, index=False)
        else:
            features.to_csv(base_path+f'angle_distance_features/{i:02d}.csv', header=None, index=False)
elif datasets_name[dataset_idx] == 'MyDataset':
    for i in range(1,13):
        print(f'processing folder {i}')
        # folder_path = glob.glob(f'{base_path}{i:02d}*/')[0]
        folder_path = f'{base_path}{i:02d}/'
        # coordinates = pd.read_csv(folder_path + f'coordinates_movnet/01.csv', header=None)
        coordinates = pd.read_csv(folder_path + f'coordinates_movnet.csv', header=None)
        features = coordinates.copy()
        n = len(coordinates)
        for j in range(n):
            feat = np.array(get_features(coordinates.iloc[j,1:].to_numpy(), feature_types[type_idx]))
            features.iloc[j,1:] = feat.reshape((-1,))
        if feature_types[type_idx] == 'array':
            features.to_csv(os.path.join(folder_path, 'array_features.csv'), header=None, index=False)
        else:
            features.to_csv(os.path.join(folder_path, 'angle_distance_features.csv'), header=None, index=False)
print('Done.')