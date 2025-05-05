import cv2 as cv
import os
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime
import uuid
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    """Configuration parameters for the license plate detection system"""
    detection_confidence: float = 0.5
    ocr_confidence: float = 0.5
    process_every_n_frames: int = 1
    target_fps: int = 20
    video_path: str = '/home/ubuntu/Videos/alprVideo.mp4'
    model_path: str = 'models/best4.pt'

class LicensePlateDetector:
    def __init__(self, config: Config):
        self.config = config
        self.reader = easyocr.Reader(['en'])
        self.model = YOLO(config.model_path)
        
    def process_video(self) -> Tuple[str, str]:
        """Main video processing pipeline"""
        video = self._setup_video()
        output_path, log_path = self._setup_output_files(video)
        
        frame_count = 0
        ret = True
        
        while ret:
            ret, frame = video.read()
            if not ret:
                break
                
            if frame_count % self.config.process_every_n_frames == 0:
                self._process_frame(frame, frame_count, video, output_path, log_path)
            
            frame_count += 1
            
        video.release()
        return output_path, log_path
    
    def _setup_video(self) -> cv.VideoCapture:
        """Initialize video capture and verify it's opened correctly"""
        video = cv.VideoCapture(self.config.video_path)
        if not video.isOpened():
            raise ValueError(f"Error: Could not open video file {self.config.video_path}")
        return video
    
    def _setup_output_files(self, video: cv.VideoCapture) -> Tuple[str, str]:
        """Setup output directories and files"""
        # Create necessary directories
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Setup video writer
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        
        output_filename = os.path.splitext(os.path.basename(self.config.video_path))[0] + '_inference_processed.avi'
        output_path = os.path.join('outputs', output_filename)
        
        self.video_writer = cv.VideoWriter(
            output_path,
            cv.VideoWriter_fourcc(*'DIVX'),
            self.config.target_fps,
            size
        )
        
        # Setup log file
        log_filename = f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = os.path.join('logs', log_filename)
        
        return output_path, log_path
    
    def _process_frame(self, frame: np.ndarray, frame_count: int, video: cv.VideoCapture, 
                      output_path: str, log_path: str) -> None:
        """Process a single frame"""
        timestamp = video.get(cv.CAP_PROP_POS_MSEC) / 1000.0
        results = self.model.track(frame, persist=True)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, confidence in zip(boxes, confidences):
            if confidence < self.config.detection_confidence:
                continue
                
            self._process_detection(frame, box, confidence, frame_count, timestamp, log_path)
        
        composed = results[0].plot()
        self.video_writer.write(composed)
    
    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhance license plate image for better OCR results"""
        # Convert to grayscale
        gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv.THRESH_BINARY, 11, 2)
        
        # Optional: denoise
        denoised = cv.fastNlMeansDenoising(thresh, h=10)
        
        return denoised
    
    def _process_detection(self, frame: np.ndarray, box: np.ndarray, confidence: float,
                         frame_count: int, timestamp: float, log_path: str) -> None:
        """Process a single detection within a frame"""
        x1, y1, x2, y2 = map(int, box[:4])
        plate_region = frame[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            return
            
        # Preprocess the plate image
        processed_plate = self._preprocess_plate(plate_region)
        
        # Convert back to 3-channel image for EasyOCR if needed
        processed_plate = cv.cvtColor(processed_plate, cv.COLOR_GRAY2BGR)
        
        ocr_result = self.reader.readtext(processed_plate)
        
        if not ocr_result:
            return
            
        text = ocr_result[0][1].upper()
        ocr_confidence = ocr_result[0][2]
        
        if ocr_confidence < self.config.ocr_confidence:
            return
            
        frame_uuid = str(uuid.uuid4())
        self._log_detection(frame_uuid, frame_count, timestamp, text, confidence, 
                          ocr_confidence, log_path)
        
        cv.putText(frame, f"{text} ({ocr_confidence:.2f})", 
                  (x1, y2+25), cv.FONT_HERSHEY_SIMPLEX,
                  1.2, (36,255,12), 3)
    
    def _log_detection(self, frame_uuid: str, frame_count: int, timestamp: float,
                      text: str, confidence: float, ocr_confidence: float, log_path: str) -> None:
        """Log a detection to file"""
        log_entry = (f"UUID: {frame_uuid}, Frame: {frame_count}, Time: {timestamp:.2f}s, "
                    f"Plate: {text}, Detection Confidence: {confidence:.2f}, "
                    f"OCR Confidence: {ocr_confidence:.2f}\n")
        
        with open(log_path, 'a') as log_file:
            log_file.write(log_entry)

def main():
    config = Config()
    detector = LicensePlateDetector(config)
    output_path, log_path = detector.process_video()
    
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Detection log saved to: {log_path}")

if __name__ == "__main__":
    main()