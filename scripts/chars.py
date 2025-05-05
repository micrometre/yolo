import cv2
import os
import time
from ultralytics import YOLO

# Configuration
TARGET_FPS = 10  # Set your desired processing FPS (set to None for original FPS)
SHOW_PREVIEW = False  # Set to True to show real-time preview
MIN_VEHICLE_SCORE = 0.5
MIN_PLATE_SCORE = 0.3

# Initialize models
coco_model = YOLO('models/yolov8s.pt')  # Vehicle detection
np_model = YOLO('models/best4.pt')      # License plate detection

# Video setup
video_path = os.path.expanduser('~/Videos/alprVideo.mp4')
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise ValueError(f"Could not open video: {video_path}")

# Video properties
original_fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_delay = 1/original_fps if TARGET_FPS is None else 1/TARGET_FPS
processing_fps = original_fps if TARGET_FPS is None else TARGET_FPS

print(f"Original video: {original_fps:.2f} FPS, {total_frames} frames")
print(f"Processing at: {processing_fps:.2f} FPS ({(processing_fps/original_fps)*100:.1f}% of original)")

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Processing variables
vehicles = [2, 3, 5]  # COCO class IDs for cars, motorcycles, buses
frame_number = 0
plates_detected = 0
start_time = time.time()
last_frame_time = start_time

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # FPS control - skip frames if needed
    if TARGET_FPS is not None:
        elapsed = current_time - last_frame_time
        if elapsed < frame_delay:
            frame_number += 1
            continue
        last_frame_time = current_time
    
    # Progress update
    if frame_number % 100 == 0:
        elapsed_time = current_time - start_time
        remaining_frames = total_frames - frame_number
        estimated_remaining = (elapsed_time/frame_number) * remaining_frames if frame_number > 0 else 0
        print(f"Frame {frame_number}/{total_frames} ({frame_number/total_frames*100:.1f}%) | "
              f"Plates: {plates_detected} | "
              f"Elapsed: {elapsed_time:.1f}s | "
              f"Remaining: {estimated_remaining:.1f}s")

    # Vehicle detection
    detections = coco_model(frame)[0]  # Using detect instead of track for stability
    
    for detection in detections.boxes.data.tolist():
        # Handle both detection formats
        if len(detection) == 6:
            x1, y1, x2, y2, score, class_id = detection
            track_id = frame_number  # Fallback ID
        else:
            x1, y1, x2, y2, track_id, score, class_id = detection
        
        if int(class_id) in vehicles and score > MIN_VEHICLE_SCORE:
            # Get vehicle ROI
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # License plate detection
            license_plates = np_model(roi)[0]
            
            for i, plate in enumerate(license_plates.boxes.data.tolist()):
                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = plate
                
                if plate_score > MIN_PLATE_SCORE:
                    plate_img = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                    
                    if plate_img.size > 0:
                        plate_filename = f"{output_dir}/plate_{track_id}_{frame_number}_{i}.jpg"
                        cv2.imwrite(plate_filename, plate_img)
                        plates_detected += 1
                        print(f"Frame {frame_number}: Vehicle {track_id} - Plate {i} - Score: {plate_score:.2f}")
    
    # Optional preview
    if SHOW_PREVIEW:
        preview = cv2.resize(frame, (1280, 720))
        cv2.imshow('Processing Preview', preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    frame_number += 1

# Cleanup
video.release()
if SHOW_PREVIEW:
    cv2.destroyAllWindows()

# Performance stats
processing_time = time.time() - start_time
actual_fps = frame_number / processing_time
print(f"\nProcessing complete!")
print(f"Processed {frame_number} frames in {processing_time:.1f} seconds")
print(f"Actual processing FPS: {actual_fps:.2f}")
print(f"License plates detected: {plates_detected}")
print(f"Results saved in: {os.path.abspath(output_dir)}")