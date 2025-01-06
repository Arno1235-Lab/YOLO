from ultralytics import settings, YOLO

# Update a setting
settings.update({"mlflow": True})

# Load a model
model = YOLO("yolov8m-seg.pt")

# Train the model
train_results = model.train(
    project="001-IS-MVTec", # mlflow experiment name
    data="./data/dataset/dataset.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    batch=2,    # batch size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

