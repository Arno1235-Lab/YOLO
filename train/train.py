import sys
sys.path.append('..')
from yolo_utils import set_mlflow_tracking, parse_config
from ultralytics import YOLO


if __name__ == '__main__':
    set_mlflow_tracking()

    config_file = parse_config()


    # Load a model
    model = YOLO(config_file['model']['path'])

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

