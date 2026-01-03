#!/usr/bin/env python3
"""
Refactored Object Detection Pipeline for Video using YOLOv8
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO


def setup_yolov8_detector(model_path="best4.pt", confidence_threshold=0.5, device="cpu"):
    """Set up YOLOv8 predictor for object detection."""
    model = YOLO(model_path)
    
    # Set model parameters
    model.conf = confidence_threshold
    model.iou = 0.45 
    
    # Determine device
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    model.to(device)
    return model


def process_frame_yolo(frame, model, objects_only=False):
    """
    Process a single frame using YOLOv8.
    Returns annotated frame and detection list.
    """
    results = model(frame, verbose=False)
    detections = []
    
    # Extract results from the first (and only) image in the batch
    result = results[0]
    
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            detections.append({
                'box': box,
                'class': model.names[cls_id],
                'score': conf,
                'class_id': cls_id
            })
    
    if objects_only and len(detections) == 0:
        return None, []
    
    # Returns BGR frame by default if input was BGR
    annotated_frame = result.plot() 
    
    return annotated_frame, detections


def process_video_yolo(video_path, output_dir="output", 
                       frame_skip=1, threshold=0.5, device="cpu", 
                       save_video=True, objects_only=False,
                       model_path="best4.pt"):
    """Process video file frame by frame using YOLOv8."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing: {video_path.name} | Model: {model_path} | Device: {device}")
    
    model = setup_yolov8_detector(model_path, threshold, device)
    
    video_writer = None
    if save_video:
        output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc="Processing Video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # Note: We pass frame (BGR) directly. YOLOv8 handles the conversion internally.
            annotated_frame, detections = process_frame_yolo(frame, model, objects_only)
            
            if annotated_frame is not None:
                # Save individual frame
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.jpg"), annotated_frame)
                saved_count += 1
                
                if video_writer is not None:
                    video_writer.write(annotated_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if video_writer:
        video_writer.release()
    
    print(f"\nComplete! Saved {saved_count} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Video Processor")
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("-s", "--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("-d", "--device", default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("-m", "--model", default="models/best4.pt", help="Path to YOLO model")
    parser.add_argument("--objects-only", action="store_true", help="Only save frames with detections")
    parser.add_argument("--no-video", action="store_false", dest="save_video", help="Disable video saving")
    
    args = parser.parse_args()
    
    process_video_yolo(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        threshold=args.threshold,
        device=args.device,
        save_video=args.save_video,
        objects_only=args.objects_only,
        model_path=args.model
    )


if __name__ == "__main__":
    main()