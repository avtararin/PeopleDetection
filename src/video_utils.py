import cv2
import torch

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap
def write_video(output_path, frame_width, frame_height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out
def draw_detections(frame, detections):
    for *box, conf, cls in detections:
        if cls == 0:
            label = f'person{conf:.2}'
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame