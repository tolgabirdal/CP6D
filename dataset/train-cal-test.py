import pandas as pd
import numpy as np

# Load the data
all_scenes_df = pd.read_csv('/home/runyi/Project/CP6D/dataset/7Scenes_0.5/7scenes_all_scenes.csv_results.csv')
subset_df = pd.read_csv('/home/runyi/Project/CP6D/dataset/7Scenes_0.5/7scenes_all_scenes.csv_0.5_subset.csv')

# Get the !subset
index = all_scenes_df['img_path'].isin(subset_df['img_path'])
calibration_set = all_scenes_df[index]

print(calibration_set.shape)
