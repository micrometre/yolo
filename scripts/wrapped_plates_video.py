#!/usr/bin/env python3
import cv2
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def setup_models(vehicle_model_path, plate_model_path, threshold, device):
    """Initialize both vehicle and license plate models."""
    v_model = YOLO(vehicle_model_path)
    p_model = YOLO(plate_model_path)
    
    # Auto-detect CUDA
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    
    v_model.to(device)
    p_model.to(device)
    
    # Set shared thresholds
    v_model.conf = threshold
    p_model.conf = threshold
    
    return v_model, p_model

def process_frame_two_stage(frame, v_model, p_model, vehicle_classes, output_dir, frame_idx):
    """
    Two-stage detection: 
    1. Detect Vehicles -> 2. Detect Plates in Vehicle ROI -> 3. Save Plate Crop
    """
    # Stage 1: Vehicle Detection
    v_results = v_model(frame, verbose=False)[0]
    plates_in_frame = 0
    
    # Iterate through detected objects
    for v_det in v_results.boxes.data.tolist():
        # Handle different YOLO result formats (6 or 7 elements)
        if len(v_det) == 6:
            vx1, vy1, vx2, vy2, v_score, v_class_id = v_det
        else:
            vx1, vy1, vx2, vy2, _, v_score, v_class_id = v_det

        # Check if the object is a vehicle (car, motorcycle, bus)
        if int(v_class_id) in vehicle_classes:
            # Create Vehicle ROI
            roi = frame[int(vy1):int(vy2), int(vx1):int(vx2)]
            if roi.size == 0:
                continue

            # Stage 2: License Plate Detection inside Vehicle ROI
            p_results = p_model(roi, verbose=False)[0]
            
            for i, p_det in enumerate(p_results.boxes.data.tolist()):
                px1, py1, px2, py2, p_score, _ = p_det
                
                # Adaptation of lines 81-94: Save cropped license plate
                plate_img = roi[int(py1):int(py2), int(px1):int(px2)]
                if plate_img.size > 0:
                    plate_filename = output_dir / f"plate_f{frame_idx}_v{int(v_class_id)}_{i}.jpg"
                    cv2.imwrite(str(plate_filename), plate_img)
                    plates_in_frame += 1

    # Return annotated frame for video output
    return v_results.plot(), plates_in_frame

def process_video_alpr(video_path, output_dir, vehicle_model, plate_model, 
                       frame_skip=1, threshold=0.5, device="cpu", save_video=True):
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    v_model, p_model = setup_models(vehicle_model, plate_model, threshold, device)
    vehicle_classes = [2, 3, 5] # car, motorcycle, bus
    
    video_writer = None
    if save_video:
        out_path = output_dir / f"{video_path.stem}_processed.mp4"
        video_writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    total_plates = 0
    pbar = tqdm(total=total_frames, desc="ALPR Processing")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % frame_skip == 0:
            annotated_frame, plate_count = process_frame_two_stage(
                frame, v_model, p_model, vehicle_classes, output_dir, frame_idx
            )
            total_plates += plate_count
            
            if video_writer:
                video_writer.write(annotated_frame)
        
        pbar.update(1)

    cap.release()
    if video_writer: video_writer.release()
    pbar.close()
    print(f"\nProcessing Complete. Plates Saved: {total_plates}")

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Vehicle & Plate Detector")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--v-model", default="models/yolov8s.pt", help="Vehicle model (COCO)")
    parser.add_argument("--p-model", default="models/best4.pt", help="Plate model")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("-s", "--skip", type=int, default=1, help="Frame skip")
    
    args = parser.parse_args()
    
    process_video_alpr(
        args.video, args.output, args.v_model, args.p_model,
        frame_skip=args.skip, threshold=args.threshold
    )

if __name__ == "__main__":
    main()