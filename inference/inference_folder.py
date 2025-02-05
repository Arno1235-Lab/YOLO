import sys
sys.path.append('..')
from yolo_utils import set_mlflow_tracking, parse_config, get_mlflow_weights
from ultralytics import YOLO


if __name__ == '__main__':
    set_mlflow_tracking()

    config_file, args = parse_config([
        {
            'name': 'folder',
            'type': str,
            'required': True,
            'help': 'folder for inference',
        },
    ])


    # Load a model
    weight_path = get_mlflow_weights(config_file)
    model = YOLO(weight_path)

    # Predict on folder
    model.predict(
        args.folder,
        project=config_file['mlflow']['project'],
        save=True,
    )

    # TODO: log result to mlflow artifact? or does this automatically happen?
