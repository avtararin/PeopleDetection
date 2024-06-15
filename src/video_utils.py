import cv2
import torch
import argparse
from ultralytics import YOLO


class VideoManipulator:
    """
    The class is designed for importing and recording videos
    """

    def __init__(self):
        self.cap = None
        self.out = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None

    def read_video(self, video_path: str):
        """
        Read video into cap from the video_path.

        Args:
            video_path (str): Path to the video storage.
        Returns:
            cv2.VideoCapture: Capture of the video.
        """
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def write_video(self, output_path: str):
        """
        Write processed video.

        Args:
            output_path (str): The path to save the processed video
        Returns:
            cv2.VideoWriter: Item for writing frames into.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def write_frame(self, frame):
        """
        Write frames into out.
            Args:
            frame : original frame
        """
        self.out.write(frame)


class Detector:
    """
    The class is designed to upload a model and detect people on video
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load Yolo model from repository.
        """
        self.model = torch.hub.load('ultralytics/yolov5', model='yolov5s')
        #self.model = YOLO("yolov8n.pt")

    @staticmethod
    def draw_detections(frame, detections):
        """
        Draw bboxes of the people on the original frame.

        Args:
            frame : frame of the original video.
            detections : the result of the model for detecting objects on the frame.
        Returns:
            frame: New frame with bboxes.
        """
        for *box, conf, cls in detections:
            if cls == 0:
                label = f'person{conf: .2}'
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


def create_parser():
    """
    Function creates parser for command line arguments.

    Returns:
        parser: Object for parse argument values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default='../data/crowd.mp4')
    parser.add_argument('-o', '--output_path', default='../data/crowd_res.mp4')

    return parser