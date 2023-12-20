import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv

# For Images

class ObjectDetector:

    def __init__(self, img):

        self.labels = None
        self.url = img

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
        #print(results)
        return results

    def plot_bboxes(self, results, frame):

        # Extract detections (Can loop here for multiple images)
        
        #print(results)
        result = results[0]                 #Since only one image is in input so len of results = 1

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
        dframe = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return dframe

    def __call__(self):

        frame = cv2.imread(self.url)
        frame = cv2.resize(frame, (1280,720))
       
        results = self.predict(frame)

        dframe = self.plot_bboxes(results, frame)

        cv2.imshow('YOLOv8 Object Detection', dframe)

        cv2.waitKey(0)  # Will wait for infinity so close window to close detector

# main()

detector = ObjectDetector(img="static/img.jpg")  # insert image path/link
detector()
