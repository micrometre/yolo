from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# Configure Tesseract path (Windows users need this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO('models/best.pt')

def preprocess_plate(plate_img):
    """Enhance license plate image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional: denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    
    return denoised

def recognize_plate(plate_img):
    """Perform OCR on license plate image"""
    # Preprocess
    processed = preprocess_plate(plate_img)
    
    # Configure Tesseract parameters
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Run OCR
    text = pytesseract.image_to_string(processed, config=config)
    
    # Post-process text
    text = ''.join(e for e in text if e.isalnum()).upper()
    
    return text

# Process image
img = cv2.imread('images/test1.jpg')
results = model(img)

# Extract and recognize plates
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_img = img[y1:y2, x1:x2]
        
        # Recognize plate text
        plate_text = recognize_plate(plate_img)
        
        # Draw results
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('License Plate Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()