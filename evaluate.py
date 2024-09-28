from ultralytics import YOLO

# Load a model
model = YOLO("runs/segment/train2/weights/best.pt")

# Predict on folder
model.predict('dataset/images/train', save=True)
