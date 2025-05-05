import cv2 as cv
import easyocr
import matplotlib.pyplot as plt
import os
from pathlib import Path
import csv
from datetime import datetime

def process_image(image, reader):
    # Convert to RGB (EasyOCR expects RGB format)
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Detect and read text
    results = reader.readtext(rgb_image)
    
    detected_items = []
    # Draw results on image and collect text
    for (bbox, text, prob) in results:
        # Convert bbox points to integers
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        br = tuple(map(int, br))
        
        # Draw rectangle and text
        cv.rectangle(image, tl, br, (0, 255, 0), 2)
        cv.putText(image, text, (tl[0], tl[1] - 10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        print(f"Detected Text: {text} (Confidence: {prob:.2f})")
        detected_items.append({
            'text': text,
            'confidence': prob,
            'position': f"{tl}-{br}"
        })
    
    return image, detected_items

def process_files(input_dir):
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'detection_results_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'filename', 'text', 'confidence', 'position']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process all files in the directory
        for file_path in Path(input_dir).glob('*'):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                if extension in ['.jpg', '.jpeg', '.png']:
                    # Process image file
                    print(f"Processing image: {file_path}")
                    image = cv.imread(str(file_path))
                    if image is not None:
                        processed_image, detected_items = process_image(image, reader)
                        output_path = os.path.join(output_dir, f"processed_{file_path.name}")
                        cv.imwrite(output_path, processed_image)
                        
                        # Write results to CSV
                        for item in detected_items:
                            writer.writerow({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': file_path.name,
                                'text': item['text'],
                                'confidence': item['confidence'],
                                'position': item['position']
                            })
                        
                elif extension == '.mp4':
                    # Process video file
                    print(f"Processing video: {file_path}")
                    cap = cv.VideoCapture(str(file_path))
                    
                    # Get video properties
                    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv.CAP_PROP_FPS))
                    
                    # Create video writer
                    output_path = os.path.join(output_dir, f"processed_{file_path.name}")
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                    
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_count += 1
                        if frame_count % 30 == 0:  # Process every 30th frame
                            processed_frame, detected_items = process_image(frame, reader)
                            out.write(processed_frame)
                            
                            # Write results to CSV
                            for item in detected_items:
                                writer.writerow({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'filename': f"{file_path.name}_frame_{frame_count}",
                                    'text': item['text'],
                                    'confidence': item['confidence'],
                                    'position': item['position']
                                })
                        else:
                            out.write(frame)
                    
                    cap.release()
                    out.release()
    
    print(f"All files processed. Results saved in: {output_dir}")
    print(f"CSV results saved to: {csv_path}")

if __name__ == "__main__":
    # Directory containing images and videos
    input_dir = "images"
    process_files(input_dir)

