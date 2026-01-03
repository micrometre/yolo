import glob
import cv2 as cv
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime


# Input video path
# read in video paths
VIDEO_PATH = 'videos/alpr.mp4'  # Change this to your video file name
# pick pre-trained model
model_pretrained = YOLO('models/best.pt')

# read video by index
video = cv.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise ValueError(f"Error: Could not open video file {VIDEO_PATH}")



# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Create output directory if it doesn't exist
os.makedirs('./outputs', exist_ok=True)  # This line creates the directory

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./outputs/uk_dash_2.avi', fourcc, 20.0, size)

# read frames
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        # detect & track objects
        results = model_pretrained.track(frame, persist=True)

        # plot results
        composed = results[0].plot()

        # save video
        out.write(composed)

out.release()
video.release()