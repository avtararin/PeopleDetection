import cv2
import sys
from video_utils import VideoManipulator, Detector, create_parser


def main(video_path, output_path):
    video_man = VideoManipulator()
    detector = Detector()

    video_man.read_video(video_path)
    video_man.write_video(output_path)
    detector.load_model()

    if not video_man.cap.isOpened():
        print("Error opening video stream file")
        return

    video_man.write_video(output_path)
    while video_man.out.isOpened():
        ret, frame = video_man.cap.read()
        if not ret:
            break
        result = detector.model(frame)
        frame = detector.draw_detections(frame, result.xyxy[0])
        video_man.write_frame(frame)
    video_man.cap.release()
    video_man.out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])
    main(namespace.video, namespace.output_path)

