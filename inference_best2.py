import cv2 as cv
import os
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime
import uuid

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Add OCR confidence threshold parameter
DETECTION_CONFIDENCE = 0.7  # Minimum confidence threshold for license plate detection (0.0 to 1.0)
OCR_CONFIDENCE = 0.5       # Minimum confidence threshold for OCR text recognition (0.0 to 1.0)

# Input video path
VIDEO_PATH = '/home/ubuntu/Videos/alprVideo1.mp4'  # Change this to your video file name

# Add these parameters after VIDEO_PATH
PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame
TARGET_FPS = 35  # Target frames per second

# pick pre-trained model
model_pretrained = YOLO('../models/best2.pt')

# read video
video = cv.VideoCapture(VIDEO_PATH)
original_fps = video.get(cv.CAP_PROP_FPS)

if not video.isOpened():
    raise ValueError(f"Error: Could not open video file {VIDEO_PATH}")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Define output path
output_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0] + 'inference_processed.avi'
output_path = os.path.join('outputs', output_filename)

# Modify the VideoWriter fps to match target fps
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter(output_path, fourcc, TARGET_FPS, size)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create log file with timestamp in name
log_filename = f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join('logs', log_filename)

# read frames
ret = True

# Initialize frame counter
frame_count = 0

while ret:
    ret, frame = video.read()
    
    if ret:
        # Only process every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            timestamp = video.get(cv.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            
            # detect & track objects
            results = model_pretrained.track(frame, persist=True)
            
            # Get bounding boxes and confidence scores
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Process each detected license plate
            for box, confidence in zip(boxes, confidences):
                # Skip if detection confidence is below threshold
                if confidence < DETECTION_CONFIDENCE:
                    continue
                    
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Extract license plate region
                plate_region = frame[y1:y2, x1:x2]
                
                # Read text from the license plate
                if plate_region.size > 0:  # Check if region is not empty
                    ocr_result = reader.readtext(plate_region)
                    
                    if ocr_result:
                        # Get the text with highest confidence
                        text = ocr_result[0][1].upper()
                        ocr_confidence = ocr_result[0][2]
                        print((ocr_result))
                        
                        # Skip if OCR confidence is below threshold
                        if ocr_confidence < OCR_CONFIDENCE:
                            continue
                        
                        # Generate UUID for this frame
                        frame_uuid = str(uuid.uuid4())
                        
                        # Create log entry with UUID
                        log_entry = f"UUID: {frame_uuid}, Frame: {frame_count}, Time: {timestamp:.2f}s, Plate: {text}, Detection Confidence: {confidence:.2f}, OCR Confidence: {ocr_confidence:.2f}\n"
                        
                        # Write to log file
                        with open(log_path, 'a') as log_file:
                            log_file.write(log_entry)
                        
                        # Draw text below the bounding box
                        cv.putText(frame, f"{text} ({ocr_confidence:.2f})", 
                                 (x1, y2+25), cv.FONT_HERSHEY_SIMPLEX,
                                 1.2, (36,255,12), 3)

            # plot results
            composed = results[0].plot()

            # save video
            out.write(composed)
            
        frame_count += 1

out.release()
video.release()
print(f"Processing complete. Output saved to: {output_path}")
print(f"Detection log saved to: {log_path}")