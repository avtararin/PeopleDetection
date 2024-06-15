#import cv2
#from ultralytics import YOLO
# video_path = '../data/crowd.mp4'
# cap = cv2.VideoCapture(video_path)
# print(type(cap))

# model = YOLO("yolov8n.pt")
#
# results = model("../data/26442601.jpg")
# print(result.boxes)

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("../data/26442601.jpg")
print(results[0])