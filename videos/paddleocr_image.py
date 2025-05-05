from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR
# For English plates:
ocr = PaddleOCR(use_angle_cls=True, lang='en')  
# For multilingual plates:
# ocr = PaddleOCR(use_angle_cls=True, lang='ml')  # 'ml' = multilingual

# Load YOLOv8 license plate detector
model = YOLO('models/best4.pt')

def preprocess_plate(plate_img):
    """Enhance license plate image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10)
    
    return denoised

def recognize_plate_paddle(plate_img):
    """Perform OCR using PaddleOCR"""
    # Preprocess
    processed = preprocess_plate(plate_img)
    
    # Run PaddleOCR
    result = ocr.ocr(processed, cls=True)
    
    # Extract text
    plate_text = ""
    if result and result[0]:
        for line in result[0]:
            if line and line[1]:
                plate_text += line[1][0] + " "
    
    # Post-process text
    plate_text = ''.join(e for e in plate_text if e.isalnum()).upper()
    
    return plate_text.strip()

# Process image
img = cv2.imread('images/test2.jpg')
results = model(img)

# Extract and recognize plates
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_img = img[y1:y2, x1:x2]
        
        # Recognize plate text
        plate_text = recognize_plate_paddle(plate_img)
        
        # Draw results
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('License Plate Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()