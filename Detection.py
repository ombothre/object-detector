import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv

class ObjectDetector:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        xyxys = []              #box coordinates
        confidences = []
        class_ids = []
        objects = []
        
        # Extract detections
        for result in results:
            boxes = result.boxes.cpu().numpy()
            count = 0

            #Optional for display
            for i in range(len(boxes.conf)):

                if boxes.conf[i] > 0.8:     #object confidence > 80%

                    count+=1

                    print("\nObject ",count," : ")

                    class_id = int(boxes.cls[i])
                    print("class id :",class_id)

                    object = self.CLASS_NAMES_DICT[class_id]        #Refer coco.names for objects
                    print("object : ",object)

                    conf = boxes.conf[i]
                    print("confidence :",conf)

                    xyxy = boxes.xyxy[i]
                    print("xyxyx : ",xyxy)

                    #tracker_id = None

                    objects.append(object)

                    if (result.boxes.conf[i].cpu().numpy()) > 0.8:      #object confidence > 80%
                        xyxys.append(result.boxes.xyxy[i].cpu().numpy())
                        confidences.append(result.boxes.conf[i].cpu().numpy())
                        class_ids.append(result.boxes.cls[i].cpu().numpy().astype(int))
            
            print("Total objects detected = ",count)

            print("-----------------------------------------------------------------------------") #time loop

        # Setup detections for visualization
        
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    mask=None,
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    tracker_id=None
                    )
        #print(detections)

        # Format custom labels
        self.labels = []

        for i in detections:
            self.labels.append(f"{self.CLASS_NAMES_DICT[i[3]]} {i[2]*100:0.2f}")
        
        # Annotate and display frame
        frrame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frrame
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        #assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        succesfull = 1
        while succesfull:
          
            start_time = time()
            
            succesfull, frame = cap.read() #1st -if succesfull, frame gives a small frame(height same,width small) of image
            assert succesfull              #check by cv2.imshow(frame)
            
            frame = cv2.resize(frame, (1280, 720)) #resize frame (x-axis,y-axis), comment out for original size

            results = self.predict(frame)

            framee = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(framee, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 1)
            
            cv2.imshow('YOLOv8 Object Detection', framee)

            if cv2.waitKey(2) & 0xFF == ord('q'): #press 'q' to quit after 2 seconds
                break
        
        cap.release()   
        cv2.destroyAllWindows()

#main()
        
detector = ObjectDetector(capture_index = 0)  #0 for webcam or add video path/link as string
detector()