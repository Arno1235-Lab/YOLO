from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")

# Train the model
train_results = model.train(
    data="dataset/dataset.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Perform object detection on an image
results = model("dataset/images/val/3_012.png")
results[0].show()
