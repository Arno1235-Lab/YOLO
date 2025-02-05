import sys
sys.path.append('..')
from yolo_utils import set_mlflow_tracking, parse_config, get_mlflow_weights
from ultralytics import YOLO


if __name__ == '__main__':
    set_mlflow_tracking()

    config_file = parse_config()


    # Load a model
    weight_path = get_mlflow_weights(config_file)
    model = YOLO(weight_path)

    # TODO: run validation on test set

