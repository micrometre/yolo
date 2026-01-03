from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time




class LPRSystem:
    def __init__(self, yolo_model_path: str = "models/best2.pt", use_gpu: bool = False):
        # Initialize YOLOv8 License Plate Detector
        self.detector = YOLO(yolo_model_path)  # Load custom-trained model
        
        # Initialize PaddleOCR (for text recognition)
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Auto-detect text orientation
            lang="en",           # 'en' for English, 'ml' for multilingual
            use_gpu=use_gpu,      # Enable GPU if available
            show_log=True        # Disable logs for cleaner output
        )


    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhances license plate image for OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        return denoised        
    def recognize_plate(self, plate_img: np.ndarray) -> str:
        """Extracts text from a license plate using PaddleOCR."""
        # Preprocess image
        processed = self.preprocess_plate(plate_img)
        
        # Run PaddleOCR
        result = self.ocr.ocr(processed, cls=True)
        
        # Extract and format text
        plate_text = ""
        if result and result[0]:
            for line in result[0]:
                if line and line[1]:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.6:  # Filter low-confidence detections
                        plate_text += text + " "
        
        # Post-process text (remove special chars, format)
        plate_text = ''.join(c for c in plate_text if c.isalnum()).upper()
        
        return plate_text.strip()        


    def process_image(self, img_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Processes a single image and returns annotated result."""
        img = cv2.imread(img_path)
        results = self.detector(img)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = img[y1:y2, x1:x2]
                
                # Recognize plate text
                plate_text = self.recognize_plate(plate_img)
                
                # Draw bounding box and text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img


    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fps: int = 30,
        show_live: bool = True
    ):
        """Processes a video file or live camera feed."""
        cap = cv2.VideoCapture(video_path)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect license plates
            results = self.detector(frame)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Recognize text
                    plate_text = self.recognize_plate(plate_img)
                    
                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
            
            if output_path:
                out.write(frame)
            
            if show_live:
                cv2.imshow("License Plate Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


lpr = LPRSystem(use_gpu=True)  # Enable GPU if available
lpr.process_video("videos/alprVideo.mp4", "output.mp4", show_live=True)