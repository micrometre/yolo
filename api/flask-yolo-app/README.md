# README.md content

# Flask YOLO Object Detection App

This project is a Flask application that allows users to upload MP4 video files for object detection using the YOLO (You Only Look Once) model. The application processes the uploaded video and displays the results.

## Project Structure

- `src/app.py`: Entry point of the Flask application.
- `src/utils/detector.py`: Utility functions for loading the YOLO model and processing videos.
- `src/templates/index.html`: HTML template for the upload form.
- `src/templates/result.html`: HTML template for displaying detection results.
- `src/static/css/style.css`: CSS styles for the application.
- `requirements.txt`: Lists the dependencies required for the application.
- `models/yolov8s.pt`: Pre-trained YOLO model for object detection.
- `uploads/`: Directory for temporarily storing uploaded MP4 files.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-yolo-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python src/app.py
   ```

4. Open your web browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

- Use the upload form to select and upload an MP4 video file.
- The application will process the video and display the results on a new page.

## License

This project is licensed under the MIT License.