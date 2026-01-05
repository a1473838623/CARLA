import os
import subprocess
import pandas as pd

csv_reader = pd.read_csv('datasets/MSL_SMAP/labeled_anomalies.csv')

data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']

for file_name in data_info['chan_id']:
    print(file_name)
    subprocess.run([
        'python', 'carla_pretext.py',
        '--config_env', 'configs/env.yml',
        '--config_exp', 'configs/pretext/carla_pretext_msl.yml',
        '--fname', file_name
    ])
    subprocess.run([
        'python', 'carla_classification.py',
        '--config_env', 'configs/env.yml',
        '--config_exp', 'configs/classification/carla_classification_msl.yml',
        '--fname', file_name
    ])