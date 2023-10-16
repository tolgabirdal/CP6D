import numpy as np
import pandas as pd
from IPython import embed
from tqdm import tqdm
import os

def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return None

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:  
		os.makedirs(path) 
		print("---  new folder...  ---")
		print("---  OK  ---")
	else:
		print("---  There is this folder!  ---")

def read_labels_file(labels_file):
    df = pd.read_csv(labels_file)
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return scene_name_to_id, poses, scenes, scenes_ids


p = 0.5
dataset = "Cambridge"
labels_file = "Train-Cal/Cambridge/cambridge_four_scenes.csv"
np.random.seed(709)
mkdir("Train-Cal/Cambridge/train_cal_{}".format(p))

scene_name_to_id, poses, scenes, scenes_ids = read_labels_file(labels_file)
train_indices = []
calib_indices = []
df = pd.read_csv(labels_file)
for scene_id in np.unique(scenes_ids):
    scene_indices = np.where(scenes_ids == scene_id)[0]
    scene_indices = np.random.permutation(len(scene_indices))
    train_index = scene_indices[:int(scene_indices.size * p)]
    calib_index = scene_indices[int(scene_indices.size * p):]
    
    split_scene_train_df = df.iloc[train_index, :]
    split_scene_train_df.to_csv("Train-Cal/Cambridge/train_cal_{}/Cambridge_subset_{}_train.csv".format(p, get_key(scene_name_to_id, scene_id)), index=False)
    split_scene_calib_df = df.iloc[calib_index, :]
    split_scene_calib_df.to_csv("Train-Cal/Cambridge/train_cal_{}/Cambridge_subset_{}_calib.csv".format(p, get_key(scene_name_to_id, scene_id)), index=False)
    train_indices.append(train_index)
    calib_indices.append(calib_index)
    print("Scene: {}, Train: {}, Calib: {}".format(get_key(scene_name_to_id, scene_id), len(train_index), len(calib_index)))
train_indices = np.concatenate(train_indices)
calib_indices = np.concatenate(calib_indices)

df_train = df.iloc[train_indices, :]
df_calib = df.iloc[calib_indices, :]
df_train.to_csv("Train-Cal/Cambridge/train_cal_{}/Cambridge_train.csv".format(p), index=False)
df_calib.to_csv("Train-Cal/Cambridge/train_cal_{}/Cambridge_calib.csv".format(p), index=False)
print("Train: {}, Calib: {}".format(len(df_train), len(df_calib)))