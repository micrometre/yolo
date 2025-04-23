from train import TrainingConfig, PlateDetectorTrainer
from app import Config, PlateDetector

def main():
    # First run detection on some images to get initial samples
    detector_config = Config()
    detector = PlateDetector(detector_config)
    detector.process_directory()
    
    # Train with the detected plates
    training_config = TrainingConfig()
    trainer = PlateDetectorTrainer(training_config)
    
    # Create training samples
    trainer.create_positive_samples(detector.plates_dir)
    trainer.create_negative_samples(detector.images_dir)
    
    # Train cascade classifier
    trainer.train_cascade()
    
if __name__ == "__main__":
    main()