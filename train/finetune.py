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

    # Train the model
    train_results = model.train(
        project=config_file['mlflow']['project'],
        data=config_file['dataset']['path'],
        epochs=config_file['train']['epochs'],
        imgsz=config_file['train']['imgsz'],
        batch=config_file['train']['batch'],
        device=config_file['train']['device'],
    )

    # TODO: log weights to mlflow artifact? or does this automatically happen?

