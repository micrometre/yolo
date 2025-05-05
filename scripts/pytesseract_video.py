from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import os
from datetime import datetime
import uuid

# Configure Tesseract path (Windows users need this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO('models/best.pt')

def basic_preprocess(plate_img):
    """Basic preprocessing for license plate images"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional: denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    
    return denoised

def advanced_preprocess(plate_img):
    """Advanced preprocessing pipeline for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Deskew (optional)
    coords = np.column_stack(np.where(cleaned > 0))
    if len(coords) > 0:  # Only deskew if we found any text regions
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = cleaned.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(cleaned, M, (w, h), 
                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return cleaned

def recognize_plate(plate_img, use_advanced=True):
    """Perform OCR on license plate image"""
    # Preprocess using selected method
    processed = advanced_preprocess(plate_img) if use_advanced else basic_preprocess(plate_img)
    
    # Configure Tesseract parameters
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Run OCR
    text = pytesseract.image_to_string(processed, config=config)
    
    # Post-process text
    text = ''.join(e for e in text if e.isalnum()).upper()
    
    return text

def setup_output_files(video_path, video):
    """Setup output directories and files"""
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('debug', exist_ok=True)  # For saving preprocessed images
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    
    output_filename = os.path.splitext(os.path.basename(video_path))[0] + '_processed.avi'
    output_path = os.path.join('outputs', output_filename)
    
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        20,  # FPS
        size
    )
    
    log_filename = f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join('logs', log_filename)
    
    return video_writer, output_path, log_path

def log_detection(log_path, frame_count, timestamp, text, confidence, debug_img=None):
    """Log a detection to file"""
    frame_uuid = str(uuid.uuid4())
    log_entry = (f"UUID: {frame_uuid}, Frame: {frame_count}, Time: {timestamp:.2f}s, "
                f"Plate: {text}, Detection Confidence: {confidence:.2f}\n")
    
    with open(log_path, 'a') as log_file:
        log_file.write(log_entry)
    
    # Optionally save debug image
    if debug_img is not None:
        debug_filename = f"debug_{frame_count}_{frame_uuid}.png"
        debug_path = os.path.join('debug', debug_filename)
        cv2.imwrite(debug_path, debug_img)

# Process video
video_path = 'videos/alprVideo.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file {video_path}")

video_writer, output_path, log_path = setup_output_files(video_path, cap)

frame_count = 0
detection_confidence_threshold = 0.5
use_advanced_preprocessing = True  # Toggle between basic and advanced preprocessing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            if confidence < detection_confidence_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            
            plate_text = recognize_plate(plate_img, use_advanced_preprocessing)
            
            if plate_text:
                # Get processed image for debugging
                processed_img = advanced_preprocess(plate_img) if use_advanced_preprocessing else basic_preprocess(plate_img)
                
                log_detection(
                    log_path, 
                    frame_count, 
                    timestamp, 
                    plate_text, 
                    float(confidence),
                    debug_img=processed_img
                )
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    video_writer.write(frame)
    
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to: {output_path}")
print(f"Detection log saved to: {log_path}")
print(f"Debug images saved to: debug/")