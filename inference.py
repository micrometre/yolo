import cv2 as cv
import os
from ultralytics import YOLO
import easyocr
import numpy as np
import csv
from datetime import datetime

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Input video path
VIDEO_PATH = 'videos/alprVideo.mp4'  # Change this to your video file name

# pick pre-trained model
model_pretrained = YOLO('models/best.pt')

# read video
video = cv.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise ValueError(f"Error: Could not open video file {VIDEO_PATH}")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Define output paths
output_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0] + 'inference_processed.avi'
output_path = os.path.join('outputs', output_filename)
csv_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0] + '_detections.csv'
csv_path = os.path.join('outputs', csv_filename)

# Initialize CSV file
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['timestamp', 'frame_number', 'license_plate', 'confidence'])

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter(output_path, fourcc, 20.0, size)

# read frames
ret = True
frame_count = 0

while ret:
    ret, frame = video.read()
    frame_count += 1

    if ret:
        # detect & track objects
        results = model_pretrained.track(frame, persist=True)
        
        # Get bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Process each detected license plate
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Extract license plate region
            plate_region = frame[y1:y2, x1:x2]
            
            # Read text from the license plate
            if plate_region.size > 0:  # Check if region is not empty
                ocr_result = reader.readtext(plate_region)
                
                if ocr_result:
                    # Get the text with highest confidence
                    text = ocr_result[0][1]
                    confidence = ocr_result[0][2]
                    
                    # Draw text below the bounding box
                    cv.putText(frame, f"{text} ({confidence:.2f})", 
                             (x1, y2+25), cv.FONT_HERSHEY_SIMPLEX,
                             0.9, (36,255,12), 2)
                    
                    # Write detection to CSV
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(csv_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([timestamp, frame_count, text, f"{confidence:.2f}"])

        # plot results
        composed = results[0].plot()

        # save video
        out.write(composed)

out.release()
video.release()
print(f"Processing complete. Output saved to:")
print(f"Video: {output_path}")
print(f"CSV: {csv_path}")