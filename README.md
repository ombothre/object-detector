# yolov8 Object Detector

This Python project implements real-time object detection using Ultralytics' YOLOv8 pretrained model. The detector identifies various objects in video streams or files, displaying labels and bounding boxes dynamically.

## Overview

This project is divided into two parts, 'img_obj-det.py' to detect objects from an image and 'vid_obj-det.py' to
detect objects from an video input or webcam setup.

The detector leverages a YOLOv8 Medium (YOLOv8m) model trained on 80 objects from the COCO dataset. It offers the ability to:

- Perform real-time object detection using webcams, connected cameras, video files, or streaming IP links.
- Display bounding boxes and labels for detected objects with their confidence levels.
- Calculate and showcase frames per second (FPS) during detection.

## Requirements

- Python (version 3.10.13)
- To install necessary dependencies, execute:
  ```bash

  pip install -r requirements.txt
- The coco.names file contains labels for 80 objects used in the YOLOv8 model.

## Usage

### Setup

- Confirm Python 3.10.13 is installed on your device.
- Install dependencies using `pip install -r requirements.txt`.
- The `coco.names` file contains labels for 80 objects used in the YOLOv8 model.

### Running the Detector

- Choose any one python file as per your requirment (image or video)
- Initialize the `ObjectDetector` class, providing:
  - A capture index (0 for webcam)
  - Video file path / image path
  - Streaming IP link
- Run the code to activate the object detection system.
- Press 'q' to exit the detection window. / Close the window for image file

### Instructions

- Users can perform real-time object detection on their device's webcam or with external cameras and video files.
- Objects detected and their confidence levels are displayed in the terminal.
- The detection window showcases labeled bounding boxes around identified objects.
- Experiment, modify, and enhance the code for custom applications.
- Feel free to explore the code and adapt it for various object detection tasks!

Enjoy detecting! 🕵️‍♂️🎥