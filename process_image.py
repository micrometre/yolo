import cv2
from ultralytics import YOLO
import pathlib
# Load a pretrained YOLOv8 model (you'd need a character recognition model)
# For this example, we'll use a general YOLOv8 model, but for character recognition,
# you'd need a custom-trained model on characters/letters
model = YOLO('models/best4.pt')  # 

# Load an image
image_path = 'test-images/anpr.png'  # Replace with your image path
image = cv2.imread(image_path)

output_path = 'output/'+image_path.split('/')[-1]  # Optional output path

# Run inference
results = model(image)

# Visualize the results
annotated_image = results[0].plot()

# Display the annotated image
cv2.imshow('YOLO Character Recognition', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



    # Save or return result
if output_path:
    # Create output directory if it doesn't exist
    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), annotated_image)
    print(f"\nResult saved to: {output_path}")


# Print detected characters (this would depend on your custom model)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    for box in boxes:
        class_id = box.cls  # Class ID (would correspond to characters in a custom model)
        conf = box.conf  # Confidence score
        xyxy = box.xyxy  # Bounding box coordinates
        print(f"Detected character: {class_id}, Confidence: {conf}, Position: {xyxy}")