from flask import Flask, request, render_template, send_file, redirect, url_for, send_from_directory
import os
from utils.detector import YOLODetector
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define absolute paths for uploads and outputs
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp4'}

# Create required directories with absolute paths
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return redirect(url_for('result', error='No video file uploaded'))
    
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('result', error='No selected file'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f'detected_{filename}'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            file.save(input_path)
            
            detector = YOLODetector()
            detector.process_video(input_path, output_path)
            
            os.remove(input_path)
            
            return redirect(url_for('result', filename=output_filename))
        except Exception as e:
            return redirect(url_for('result', error=str(e)))
    
    return redirect(url_for('result', error='Invalid file type'))

@app.route('/result')
def result():
    error = request.args.get('error')
    output_filename = request.args.get('filename')
    return render_template('result.html', error=error, output_filename=output_filename)

@app.route('/download/<filename>')
def download_video(filename):
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return render_template('result.html', error="File not found"), 404

if __name__ == '__main__':
    app.run(debug=True)