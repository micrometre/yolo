import cv2
import numpy as np
import pytesseract
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class Config:
    """Configuration settings"""
    output_dir: str = "output"
    image_dir: str = "../images"
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
    confidence_threshold: float = 0.5
    min_neighbors: int = 5
    min_plate_size: tuple = (30, 30)
    model_path: str = "../model/yolov4-tiny.weights"
    config_path: str = "yolov4-tiny.cfg"
    names_path: str = "coco.names"
    cascade_path: str = "training_data/cascade/cascade.xml"  # Path to trained cascade

class PlateDetector:
    def __init__(self, config: Config):
        self.config = config
        self.net, self.classes, self.output_layers = self._load_yolo()
        self.plate_cascade = self._load_plate_cascade()
        self._setup_directories()
        
    def _load_yolo(self):
        """Load YOLO model and configuration"""
        net = cv2.dnn.readNet(self.config.model_path, self.config.config_path)
        with open(self.config.names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers

    def _load_plate_cascade(self):
        """Load license plate cascade classifier"""
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    def _setup_directories(self):
        """Create output directories"""
        self.images_dir = os.path.join(self.config.output_dir, "images")
        self.plates_dir = os.path.join(self.config.output_dir, "plates")
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)

    def detect_objects(self, img: np.ndarray) -> Tuple[List, List, List]:
        """Detect objects in the image using YOLO"""
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.config.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids

    def _draw_labels(self, boxes: List, confidences: List, class_ids: List, img: np.ndarray):
        """Draw bounding boxes and labels on the image"""
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.config.confidence_threshold, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 2)

    def _detect_plates(self, img: np.ndarray) -> List:
        """Detect license plates in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self.config.min_neighbors, minSize=self.config.min_plate_size
        )
        return plates

    def _process_plates(self, img: np.ndarray, plates: List, image_name: str) -> List[Dict[str, Any]]:
        """Process detected plates and extract data"""
        plates_data = []
        for (x, y, w, h) in plates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "License Plate", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            
            plate_roi = img[y:y + h, x:x + w]
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            text = pytesseract.image_to_string(opening, config='--psm 7').strip()
            
            plate_filename = os.path.join(self.plates_dir, f"{os.path.splitext(image_name)[0]}_plate_{x}_{y}.jpg")
            cv2.imwrite(plate_filename, plate_roi)
            
            # Convert NumPy types to native Python types
            plates_data.append({
                "plate_text": text,
                "coordinates": {
                    "x": int(x),  # Convert np.int32 to int
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "image_file": os.path.relpath(plate_filename, self.config.output_dir),
                "timestamp": datetime.now().isoformat()
            })
        return plates_data

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and return results"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        
        # Detect and draw objects
        boxes, confidences, class_ids = self.detect_objects(img)
        self._draw_labels(boxes, confidences, class_ids, img)
        
        # Detect and process plates
        plates = self._detect_plates(img)
        plates_data = self._process_plates(img, plates, os.path.basename(image_path))
        
        # Save processed image
        output_image = os.path.join(self.images_dir, f"processed_{os.path.basename(image_path)}")
        cv2.imwrite(output_image, img)
        
        return {
            "image": os.path.basename(image_path),
            "processed_image": os.path.relpath(output_image, self.config.output_dir),
            "plates": plates_data,
            "total_plates": len(plates_data)
        }

    def process_directory(self) -> Dict[str, Any]:
        """Process all images in the directory"""
        image_files = [f for f in os.listdir(self.config.image_dir) 
                      if f.lower().endswith(self.config.image_extensions)]
        
        if not image_files:
            print("No images found in the directory!")
            return None
            
        results = {
            "total_images": 0,
            "total_plates": 0,
            "images": []
        }
        
        for image_file in image_files:
            print(f"Processing {image_file}...")
            image_path = os.path.join(self.config.image_dir, image_file)
            
            image_result = self.process_image(image_path)
            if image_result:
                results["images"].append(image_result)
                results["total_plates"] += image_result["total_plates"]
                results["total_images"] += 1
        
        # Save results
        json_file = os.path.join(self.config.output_dir, "detection_results.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
            
        print(f"Processing complete! Results saved to {json_file}")
        return results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    config = Config()
    detector = PlateDetector(config)
    detector.process_directory()

if __name__ == "__main__":
    main()