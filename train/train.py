import sys
sys.path.append('..')
from yolo_utils import set_mlflow_tracking
import argparse
import yaml
from ultralytics import YOLO


if __name__ == '__main__':
    set_mlflow_tracking()

    parser = argparse.ArgumentParser(description='YOLO train')
    parser.add_argument('-c', '--config', type=str, required=True, help='YAML config file')
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)


    # Load a model
    model = YOLO(config_file['model']['path'])

    # Train the model
    train_results = model.train(
        project=config_file['mlflow']['project'],
        data=config_file['mlflow']['data'],
        epochs=config_file['mlflow']['epochs'],
        imgsz=config_file['mlflow']['imgsz'],
        batch=config_file['mlflow']['batch'],
        device=config_file['mlflow']['device'],
    )

