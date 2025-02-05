from ultralytics import settings


def set_mlflow_tracking():
    # Update a setting
    settings.update({"mlflow": True})
