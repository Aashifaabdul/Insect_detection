from ultralytics import YOLO

# Load the model
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='data.yaml',  # path to your data.yaml file
    imgsz=640,  # image size
    epochs=100,  # number of epochs
    batch=8,  # batch size
    name='yolov8s_v8_100e'  # model name
)