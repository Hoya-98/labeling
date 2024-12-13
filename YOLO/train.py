from ultralytics import YOLO
model = YOLO("yolov10x.pt")
model.train(data="./custom.yaml", device=[0, 1], epochs=1000)