from ultralytics import settings
import argparse
import yaml


def set_mlflow_tracking():
    # Update a setting
    settings.update({"mlflow": True})


def parse_config(extra_args=None):
    parser = argparse.ArgumentParser(description='YOLO train')
    parser.add_argument('-c', '--config', type=str, required=True, help='YAML config file')

    if extra_args is not None:
        for extra_arg in extra_args:
            parser.add_argument(f'-{extra_arg['name'][0]}', f'--{extra_arg['name']}', type=extra_arg['type'], required=extra_arg['required'], help=extra_arg['help'])
    
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    
    if extra_args is None:
        return config_file
    return config_file, args


def get_mlflow_weights(config_file):
    pass
