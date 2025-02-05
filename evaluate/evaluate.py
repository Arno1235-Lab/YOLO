import sys
sys.path.append('..')
from yolo_utils import set_mlflow_tracking
import argparse
import yaml
from ultralytics import YOLO


if __name__ == '__main__':
    set_mlflow_tracking()

    parser = argparse.ArgumentParser(description='YOLO evaluate')
    parser.add_argument('-c', '--config', type=str, required=True, help='YAML config file')
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)


    # TODO: load weights from mlflow or local
    # Load a model
    model = YOLO("001-IS-MVTec/train/weights/best.pt")

    # TODO: run validation on test set

    # Predict on folder
    model.predict(
        'data/dataset/images/train',
        project="001-IS-MVTec",
        save=True,
    )

