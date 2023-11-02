from ultralytics import YOLO
import glob

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml').load('runs/detect/train/weights/best.pt')  # build from YAML and transfer weights
results = model.track(source="./shrimp_video.mp4", conf=0.0001655, show=True)  # Tracking with ByteTrack tracker

# count = 0
# for idx, video in enumerate(sorted(glob.glob("/Users/javkhlan-ochirganbat/repos/machine-learning/underwater_object/UOT100/*/*.mp4"))):
#     if "shrimp" not in video.lower():
#         continue
#     count += 1
#     if count < 1:
#         continue
#     results = model.track(source=video, show=True, tracker="bytetrack.yaml", conf=0.00018, mode="track")  # Tracking with ByteTrack tracker


# Perform tracking with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
