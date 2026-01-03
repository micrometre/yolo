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
    
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    
    v_model.to(device)
    p_model.to(device)
    
    v_model.conf = threshold
    p_model.conf = threshold
    
    return v_model, p_model

def process_frame_two_stage(frame, v_model, p_model, vehicle_classes, output_dir, frame_idx, min_plate_score):
    """
    Two-stage detection: 
    1. Detect Vehicles -> 2. Detect Plates -> 3. Save Cropped Plate & Full Annotated Frame
    """
    v_results = v_model(frame, verbose=False)[0]
    plates_in_frame = 0
    
    # We create the full-frame annotation once for the whole frame
    # This draws boxes around all vehicles and plates found in this frame
    full_annotated_frame = v_results.plot() 
    
    found_any_plate = False

    for v_det in v_results.boxes.data.tolist():
        if len(v_det) == 6:
            vx1, vy1, vx2, vy2, v_score, v_class_id = v_det
        else:
            vx1, vy1, vx2, vy2, _, v_score, v_class_id = v_det

        if int(v_class_id) in vehicle_classes:
            roi = frame[int(vy1):int(vy2), int(vx1):int(vx2)]
            if roi.size == 0:
                continue

            # Detect plates within the vehicle ROI
            p_results = p_model(roi, verbose=False)[0]
            
            # If plates are found, we also want to draw them on the 'full_annotated_frame'
            # Note: v_results.plot() only shows vehicles. To show plates on the full frame,
            # we must handle the coordinate offset.
            for i, p_det in enumerate(p_results.boxes.data.tolist()):
                px1, py1, px2, py2, p_score, _ = p_det
                
                if p_score >= min_plate_score:
                    found_any_plate = True
                    # 1. Save the tightly cropped plate (Clean/Raw)
                    plate_crop = roi[int(py1):int(py2), int(px1):int(px2)]
                    
                    if plate_crop.size > 0:
                        base_name = f"f{frame_idx}_v{int(v_class_id)}_p{i}"
                        cv2.imwrite(str(output_dir / f"{base_name}_crop.jpg"), plate_crop)
                        
                        # Draw the plate box on the full annotated frame (correcting for ROI offset)
                        # This ensures the "Full Annotated Image" has both Vehicle and Plate boxes
                        top_left = (int(vx1 + px1), int(vy1 + py1))
                        bottom_right = (int(vx1 + px2), int(vy1 + py2))
                        cv2.rectangle(full_annotated_frame, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(full_annotated_frame, f"Plate {p_score:.2f}", 
                                    (top_left[0], top_left[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        plates_in_frame += 1

    # 2. Save the Full Annotated Frame (only if at least one plate was found)
    if found_any_plate:
        cv2.imwrite(str(output_dir / f"f{frame_idx}_full_annotated.jpg"), full_annotated_frame)

    return full_annotated_frame, plates_in_frame

def process_video_alpr(video_path, output_dir, v_model_path, p_model_path, 
                       frame_skip=1, threshold=0.5, device="cpu", save_video=True):
    
    video_path = Path(video_path)
    output_dir = Path(output_dir) / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    v_model, p_model = setup_models(v_model_path, p_model_path, threshold, device)
    vehicle_classes = [2, 3, 5] 
    
    video_writer = None
    if save_video:
        out_path = Path(output_dir).parent / f"{video_path.stem}_output.mp4"
        video_writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    total_plates = 0
    pbar = tqdm(total=total_frames, desc="ALPR Processing")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % frame_skip == 0:
            annotated_frame, plate_count = process_frame_two_stage(
                frame, v_model, p_model, vehicle_classes, output_dir, frame_idx, threshold
            )
            total_plates += plate_count
            
            if video_writer:
                video_writer.write(annotated_frame)
        
        pbar.update(1)

    cap.release()
    if video_writer: video_writer.release()
    pbar.close()
    print(f"\nDone! Saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="ALPR: Save Crops + Full Annotated Frames")
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument("--v-model", default="models/yolov8s.pt", help="Vehicle model")
    parser.add_argument("--p-model", default="models/best4.pt", help="Plate model")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("-s", "--skip", type=int, default=1, help="Frame skip")
    
    args = parser.parse_args()
    
    process_video_alpr(
        args.video, args.output, args.v_model, args.p_model,
        frame_skip=args.skip, threshold=args.threshold
    )

if __name__ == "__main__":
    main()