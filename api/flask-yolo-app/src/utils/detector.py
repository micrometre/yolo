import cv2 as cv
from ultralytics import YOLO
import torch
import os

class YOLODetector:
    def __init__(self):
        # Get the absolute path to the models directory
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(base_dir, 'models', 'yolov8s.pt')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO(model_path).to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO('yolov8s').to(device)
    
    def process_video(self, input_path, output_path):
        try:
            video = cv.VideoCapture(input_path)
            if not video.isOpened():
                raise ValueError("Could not open video file")
            
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            size = (frame_width, frame_height)
            
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # Changed from DIVX to XVID
            out = cv.VideoWriter(output_path, fourcc, 20.0, size)
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                results = self.model.track(frame, persist=True)
                composed = results[0].plot()
                out.write(composed)
            
        except Exception as e:
            print(f"Error processing video: {e}")
            raise
            
        finally:
            video.release()
            out.release()