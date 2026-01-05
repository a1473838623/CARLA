python carla_pretext.py --config_env configs/env.yml --config_exp configs/pretext/carla_pretext_msl.yml --fname M-6
python carla_classification.py --config_env configs/env.yml --config_exp configs/classification/carla_classification_msl.yml --fname M-6

python carla_pretext.py --config_env configs/env.yml --config_exp configs/pretext/carla_pretext_smd.yml --fname machine-1-1
python carla_classification.py --config_env configs/env.yml --config_exp configs/classification/carla_classification_smd.yml --fname machine-1-1
