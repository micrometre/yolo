from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
import os
from datetime import datetime
import uuid
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration parameters for the license plate detection system"""
    detection_confidence: float = 0.5
    ocr_confidence: float = 0.5
    process_every_n_frames: int = 1
    target_fps: int = 35
    video_path: str = '/home/ubuntu/Videos/alprVideo4.mp4'
    model_path: str = 'models/best4.pt'
    use_gpu: bool = False

class LicensePlateDetector:
    def __init__(self, config: Config):
        self.config = config
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=config.use_gpu,
            show_log=False
        )
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
        self.video_writer.release()
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
            cv.VideoWriter_fourcc(*'XVID'),
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
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, confidence in zip(boxes, confidences):
                if confidence < self.config.detection_confidence:
                    continue
                    
                self._process_detection(frame, box, confidence, frame_count, timestamp, log_path)
        
        composed = results[0].plot()
        self.video_writer.write(composed)
    
    def _process_detection(self, frame: np.ndarray, box: np.ndarray, confidence: float,
                         frame_count: int, timestamp: float, log_path: str) -> None:
        """Process a single detection within a frame"""
        x1, y1, x2, y2 = map(int, box[:4])
        plate_region = frame[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            return
            
        # Apply preprocessing to enhance OCR accuracy
        processed_plate = self.preprocess_plate(plate_region)
        
        # Run OCR on the processed plate
        ocr_result = self.ocr.ocr(processed_plate, cls=True)
        
        if not ocr_result or not ocr_result[0]:
            return
            
        # Extract text with highest confidence
        highest_conf = 0
        text = ""
        
        for line in ocr_result[0]:
            if line and len(line) >= 2 and line[1]:
                current_text = line[1][0]
                current_conf = line[1][1]
                if current_conf > highest_conf:
                    highest_conf = current_conf
                    text = current_text
        
        if not text or highest_conf < self.config.ocr_confidence:
            return
            
        # Clean text: keep only alphanumeric chars and convert to uppercase
        text = ''.join(c for c in text if c.isalnum()).upper()
        
        frame_uuid = str(uuid.uuid4())
        self._log_detection(frame_uuid, frame_count, timestamp, text, confidence, 
                          highest_conf, log_path)
        
        cv.putText(frame, f"{text} ({highest_conf:.2f})", 
                  (x1, y2+25), cv.FONT_HERSHEY_SIMPLEX,
                  1.2, (0, 255, 0), 3)

    def _log_detection(self, frame_uuid: str, frame_count: int, timestamp: float,
                      text: str, confidence: float, ocr_confidence: float, log_path: str) -> None:
        """Log a detection to file"""
        log_entry = (f"UUID: {frame_uuid}, Frame: {frame_count}, Time: {timestamp:.2f}s, "
                    f"Plate: {text}, Detection Confidence: {confidence:.2f}, "
                    f"OCR Confidence: {ocr_confidence:.2f}\n")
        
        with open(log_path, 'a') as log_file:
            log_file.write(log_entry)
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhances license plate image for improved OCR accuracy"""
        try:
            # Convert to grayscale
            gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
            
            # Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Adaptive thresholding
            binary = cv.adaptiveThreshold(
                enhanced, 255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY, 11, 2
            )
            
            # Non-local means denoising
            denoised = cv.fastNlMeansDenoising(binary, h=10)
            
            return denoised
        except Exception as e:
            print(f"Error in preprocessing plate: {e}")
            return plate_img

if __name__ == "__main__":
    try:
        config = Config()
        detector = LicensePlateDetector(config)
        output_path, log_path = detector.process_video()
        print(f"Processing complete. Output saved to {output_path}, logs saved to {log_path}")
    except Exception as e:
        print(f"Error: {e}")