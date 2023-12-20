# Yolo v8 Object Detector

Experience real-time object detection at its finest with this Python project utilizing Ultralytics' cutting-edge YOLOv8 pretrained model. Seamlessly identifying diverse objects within images or videos, it dynamically showcases labels and bounding boxes, bringing precision and efficiency to object recognition.

## Overview

This project comprises two components: `img_obj-det.py` designed for detecting objects within images, and `vid_obj-det.py` tailored for detecting objects from either video inputs or webcam setups.

The detector leverages a YOLOv8 Medium `yolov8m` model trained on 80 objects from the COCO dataset. It offers the ability to:

- Perform real-time object detection using webcams, connected cameras, video files, or streaming IP links.
- Display bounding boxes and labels for detected objects with their confidence levels.
- Calculate and showcase frames per second (FPS) during detection.

## Screenshots

### Image Model

![image](https://github.com/ombothre/object-detector/assets/92716010/732215d9-722d-47c5-9fdd-8457a9524d24)

### Video Model

![image](https://github.com/ombothre/object-detector/assets/92716010/831821de-59ca-4fd6-90b9-e83916c10f1b)

## Requirements

- Python (version *3.10.13*)
- To install necessary python libraries, execute:

  ```bash
  pip install -r requirements.txt
- GPU is optional
- Enviorment like IDE/Text Editor etc. to run the Python file.

## Usage

### Setup

- Confirm Python 3.10.13 is installed on your device.
- Install dependencies using `pip install -r requirements.txt`.
- The `coco.names` file contains labels for 80 objects used in the YOLOv8 model.
- `static/` contains sample files and `requirements.txt`.

### Running the Object Detector

- **For Image Detection (`img_obj-det.py`):**
  1. Initialize the `ObjectDetector` class, providing:
     - Image file path
     - Image link
  2. Run the code to activate the object detection system by :
      - By pressing the run button on your envoirment or
        ```bash
        python img_obj-det.py
  3. Close the window to exit the detection.

- **For Video and Webcam Detection (`vid_obj-det.py`):**
  1. Initialize the `ObjectDetector` class, providing:
     - Video file path
     - Video link
     - Index number (0 for default webcam on your device)
  2. Run the code to activate the object detection system by :
      - By pressing the run button on your envoirment or
        ```bash
        python vid_obj-det.py
  3. Press **`q`** to exit the detection.

### Instructions

- Users can perform real-time object detection on their device's webcam or with external cameras and video files.
- Objects detected and their confidence levels are displayed in the terminal.
- The detection window showcases labeled bounding boxes around identified objects.
- Experiment, modify, and enhance the code for custom applications.
- Feel free to explore the code and adapt it for various object detection tasks!

Enjoy detecting! 🕵️‍♂️🎥

### References

- Ultralytics yolov8 documentation : https://docs.ultralytics.com/
