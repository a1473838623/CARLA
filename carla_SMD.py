import subprocess

subprocess.run([
    'python', 'carla_pretext.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/pretext/carla_pretext_smd.yml',
    '--fname', 'All'
])
subprocess.run([
    'python', 'carla_classification.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/classification/carla_classification_smd.yml',
    '--fname', 'All'
])