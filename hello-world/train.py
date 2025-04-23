import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    training_data_dir: str = "training_data"
    positive_samples_dir: str = "positive_samples"
    negative_samples_dir: str = "negative_samples"
    cascade_dir: str = "cascade"
    num_stages: int = 20
    min_hit_rate: float = 0.999
    max_false_alarm_rate: float = 0.5
    
class PlateDetectorTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [
            self.config.training_data_dir,
            os.path.join(self.config.training_data_dir, self.config.positive_samples_dir),
            os.path.join(self.config.training_data_dir, self.config.negative_samples_dir),
            os.path.join(self.config.training_data_dir, self.config.cascade_dir)
        ]:
            os.makedirs(dir_path, exist_ok=True)
            
    def create_positive_samples(self, plates_dir: str):
        """Create positive samples from detected plates"""
        pos_dir = os.path.join(self.config.training_data_dir, self.config.positive_samples_dir)
        with open(os.path.join(self.config.training_data_dir, "positives.txt"), "w") as f:
            for img_file in os.listdir(plates_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(plates_dir, img_file)
                    new_path = os.path.join(pos_dir, img_file)
                    cv2.imwrite(new_path, cv2.imread(img_path))
                    f.write(f"{new_path} 1 0 0 100 40\n")  # Adjust size as needed
                    
    def create_negative_samples(self, images_dir: str):
        """Create negative samples from non-plate regions"""
        neg_dir = os.path.join(self.config.training_data_dir, self.config.negative_samples_dir)
        with open(os.path.join(self.config.training_data_dir, "negatives.txt"), "w") as f:
            for img_file in os.listdir(images_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(images_dir, img_file)
                    new_path = os.path.join(neg_dir, f"neg_{img_file}")
                    cv2.imwrite(new_path, cv2.imread(img_path))
                    f.write(f"{new_path}\n")

    def train_cascade(self):
        """Train cascade classifier"""
        cascade_dir = os.path.join(self.config.training_data_dir, self.config.cascade_dir)
        cmd = f"""
        opencv_traincascade \
        -data {cascade_dir} \
        -vec positives.vec \
        -bg negatives.txt \
        -numPos {self._count_samples(self.config.positive_samples_dir)} \
        -numNeg {self._count_samples(self.config.negative_samples_dir)} \
        -numStages {self.config.num_stages} \
        -minHitRate {self.config.min_hit_rate} \
        -maxFalseAlarmRate {self.config.max_false_alarm_rate} \
        -w 100 -h 40
        """
        os.system(cmd)

    def _count_samples(self, dir_name: str) -> int:
        """Count number of images in directory"""
        return len([f for f in os.listdir(os.path.join(self.config.training_data_dir, dir_name)) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))])