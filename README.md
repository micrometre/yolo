
# YOLO Vehicle & License Plate Detection

This project provides a complete pipeline for vehicle and license plate detection using YOLOv8, with multiple OCR/ALPR options and a Flask web application for video uploads and detection.

## Project Structure

- `api/flask-yolo-app/` — Flask web app for uploading videos and running YOLO detection
  - `src/app.py` — Main Flask app
  - `src/utils/detector.py` — YOLO detection utilities
  - `src/templates/` — Web UI templates
  - `src/static/` — CSS and static files
  - `models/` — YOLO model weights for the web app
- `models/` — YOLOv8 model weights for scripts and experiments
- `scripts/` — Standalone scripts for ALPR and OCR:
  - `easyocr_*.py` — EasyOCR-based detection
  - `paddleocr_*.py` — PaddleOCR-based detection
  - `pytesseract_*.py` — Tesseract-based detection
  - `plates_wraped.py`, `inference_csv.py`, etc. — Batch and video processing
- `output/` — Output results and detections
- `test-images/`, `videos/` — Sample images and videos

## Key Features

- **YOLOv8-based vehicle and license plate detection**
- **Flask web app** for uploading MP4 videos and visualizing detection results
- **Multiple OCR/ALPR backends:** EasyOCR, PaddleOCR, Tesseract
- **Batch and video processing scripts** for research and automation

## Setup Instructions

1. **Clone the repository:**
	```bash
	git clone https://github.com/micrometre/yolo
	cd yolo
	```

2. **Set up Python environment:**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	# For Flask app:
	pip install -r api/flask-yolo-app/requirements.txt
	```

4. **Download or place YOLO model weights** in the `models/` directory (e.g., `yolov8s.pt`, `best.pt`, etc.).

## Running the Flask Web App

```bash
cd api/flask-yolo-app
python src/app.py
# Visit http://127.0.0.1:5000 in your browser
```

## Running Scripts

Example (EasyOCR on images):
```bash
python scripts/easyocr_images.py
```
Replace with other scripts as needed for PaddleOCR, Tesseract, or video processing.

## License

This project is licensed under the MIT License.
