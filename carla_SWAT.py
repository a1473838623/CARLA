import os
import subprocess
import pandas as pd

subprocess.run([
    'python', 'carla_pretext.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/pretext/carla_pretext_swat.yml',
    '--fname', 'All'
])
subprocess.run([
    'python', 'carla_classification.py',
    '--config_env', 'configs/env.yml',
    '--config_exp', 'configs/classification/carla_classification_swat.yml',
    '--fname', 'All'
])