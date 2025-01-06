from ultralytics import YOLO

# Load a model
model = YOLO("001-IS-MVTec/train/weights/best.pt")

# TODO: run validation on test set

# Predict on folder
model.predict(
    'data/dataset/images/train',
    project="001-IS-MVTec",
    save=True,
)

