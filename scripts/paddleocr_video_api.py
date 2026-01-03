from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
import os
import uuid  # Add UUID import for generating unique identifiers
import json  # Add JSON import for file output


class LPRSystem:
    def __init__(self, yolo_model_path: str = "models/best2.pt", use_gpu: bool = False):
        # Initialize YOLOv8 License Plate Detector
        self.detector = YOLO(yolo_model_path)  # Load custom-trained model
        
        # Initialize PaddleOCR (for text recognition)
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Auto-detect text orientation
            lang="en",           # 'en' for English, 'ml' for multilingual
            use_gpu=use_gpu,      # Enable GPU if available
            show_log=False        # Disable logs for cleaner output
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
        #plate_text = ''.join(c for c in plate_text if c.isalnum()).upper()
        plate_text = plate_text
        
        return plate_text        



    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fps: int = 30,
        show_live: bool = True
    ):
        """Processes a video file or live camera feed."""
        cap = cv2.VideoCapture(video_path)
        
        # Create directories if they don't exist
        if not os.path.exists("../goanpr/public/images"):
            os.makedirs("../goanpr/public/images")
            
        if not os.path.exists("json_results"):
            os.makedirs("json_results")
            
        # Create videos directory if it doesn't exist
        if not os.path.exists("videos_output"):
            os.makedirs("videos_output")
            
        if output_path:
            # Add videos_output directory to output path
            output_path = os.path.join("videos_output", os.path.basename(output_path))
            
            # Check if output_path ends with .mp4 and change to .avi if needed
            if output_path.lower().endswith('.mp4'):
                output_path = output_path[:-4] + '.avi'
                
            # Change to AVI format
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a copy of the original frame before any annotations
            original_frame = frame.copy()
            
            # Detect license plates
            results = self.detector(frame)
            
            # Flag to track if plate was detected in this frame
            plate_detected = False
            
            for result in results:
                for box in result.boxes:
                    plate_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Recognize text
                    plate_text = self.recognize_plate(plate_img)
                    print(plate_text)
                    
                    # Generate UUID for the filename
                    unique_id = str(uuid.uuid4())
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Save the frame with UUID in filename
                    image_filename = f"images/{unique_id}.jpg"
                    cv2.imwrite(image_filename, original_frame)
                    import requests  # Add requests import for HTTP POST

                    # Create and save JSON result file
                    json_data = {
                        "uuid": unique_id,
                        "results": [
                            {
                                "plate": plate_text.strip()
                            }
                        ]
                    }
                    


                    # Send JSON data via HTTP POST request
                    url = "http://172.17.0.1:5000/alprd"  # Replace with your API endpoint
                    headers = {"Content-Type": "application/json"}
                    try:
                        response = requests.post(url, json=json_data, headers=headers)
                        if response.status_code == 200:
                            print(f"Data successfully sent to server: {response.json()}")
                        else:
                            print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"Error occurred while sending data: {e}")

                    # Save JSON file
                    json_filename = f"json_results/{unique_id}.json"
                    with open(json_filename, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)
                    
                    # Draw results on the frame copy (for display and video output only)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    
                    # Only save one frame per detection batch
                    break
            
            if output_path:
                out.write(frame)
            
            if show_live:
                cv2.imshow("License Plate Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


# Update the output file extension to .avi
lpr = LPRSystem(use_gpu=True)  # Enable GPU if available
lpr.process_video("videos/alprVideo.mp4", "output.avi", show_live=False)