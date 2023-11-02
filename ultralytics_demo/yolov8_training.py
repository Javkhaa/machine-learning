"""
Training and fine tuning
"""
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n.pt')#.load('runs/detect/train2/weights/best.pt')  # build from YAML and transfer weights
# Train the model
model.train(data='./custom_data.yaml', epochs=10, imgsz=640, device="mps")
