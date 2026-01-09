import subprocess

from pathlib import Path
folder_path = Path('./datasets/SMD/train')

for file in folder_path.iterdir():
    if file.is_file():
        file_name = file.name
        print(file_name)
        subprocess.run([
            'python', 'carla_pretext.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/pretext/carla_pretext_smd.yml',
            '--fname', file_name
        ])
        subprocess.run([
            'python', 'carla_classification.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/classification/carla_classification_smd.yml',
            '--fname', file_name
        ])

subprocess.run([
    'python', 'Evaluation_toolkit.py',
    '--dataset', 'smd',
    '--fname', 'All',
    '--mode', 'single'
])