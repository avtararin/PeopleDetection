import cv2
from video_utils import load_model, read_video, write_video, draw_detections

def main():
    video_path = '../data/crowd.mp4'
    output_path = '../data/crowd_res.mp4'

    model = load_model()

    cap = read_video(video_path)
    if not cap.isOpened():
        print("Error opening video stream file")
        return
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = write_video(output_path, frame_width, frame_height, fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = model(frame)
        frame = draw_detections(frame, result.xyxy[0])
        out.write(frame)
    cap.realese()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()