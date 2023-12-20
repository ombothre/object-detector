import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv

# For Videos

class ObjectDetector:

    def __init__(self, capture_index):

        self.labels = None
        self.capture_index = capture_index

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=2, text_thickness=2, text_scale=1.2)

    def load_model(self):

        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model (You can use any other models of yolov8 or any)
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):

        # Extract detections (Can loop here for multiple videos)
        
        #print(results)
        result = results[0]                 #Since only one video is in input so len of results = 1

        # Setup detections for visualization

        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            mask=None,
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int),
            tracker_id=None
        )

        # Format custom labels
        self.labels = []

        for i in detections:
            self.labels.append(f"{self.CLASS_NAMES_DICT[i[3]]} {i[2] * 100:0.2f}")

        # Annotate and display frame
        frrame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frrame

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        #Resizing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        successful = 1
        while successful:

            start_time = time()

            successful, frame = cap.read()  # 1st -if successful, frame gives a small frame(height same,width small) of image
            assert successful  # check by cv2.imshow(frame)

            frame = cv2.resize(frame, (854, 480))  # resize frame (x-axis,y-axis), comment out for original size

            results = self.predict(frame)

            dframe = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(dframe, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1)

            cv2.imshow('YOLOv8 Object Detection', dframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit after 2 seconds
                break

        cap.release()
        cv2.destroyAllWindows()


# main()

#detector = ObjectDetector(capture_index = 0)  # 0 for webcam or add video path/link as string
detector = ObjectDetector(capture_index = "static/vid.mp4")
detector()
