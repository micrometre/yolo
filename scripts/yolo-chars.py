import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (you'd need a character recognition model)
# For this example, we'll use a general YOLOv8 model, but for character recognition,
# you'd need a custom-trained model on characters/letters
model = YOLO('models/best.pt')  # Replace with your character recognition model

# Load an image
image_path = 'images/test1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Visualize the results
annotated_image = results[0].plot()

# Display the annotated image
cv2.imshow('YOLO Character Recognition', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detected characters (this would depend on your custom model)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    for box in boxes:
        class_id = box.cls  # Class ID (would correspond to characters in a custom model)
        conf = box.conf  # Confidence score
        xyxy = box.xyxy  # Bounding box coordinates
        print(f"Detected character: {class_id}, Confidence: {conf}, Position: {xyxy}")