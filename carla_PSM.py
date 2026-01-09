import subprocess

subprocess.run([
    'python', 'carla_pretext.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/pretext/carla_pretext_psm.yml',
    '--fname', 'All'
])
subprocess.run([
    'python', 'carla_classification.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/classification/carla_classification_psm.yml',
    '--fname', 'All'
])

subprocess.run([
    'python', 'Evaluation_toolkit.py',
    '--dataset', 'psm',
    '--fname', 'All'
])