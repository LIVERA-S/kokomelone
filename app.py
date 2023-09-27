from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression

from yolov5.models.experimental import attempt_load



app = Flask(__name__)

app.template_folder = 'templates'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLOv5 model
weights_path = 'weights/yolov5s.pt'   # Specify the path to your YOLOv5 model weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
model.eval()


def detect_damage(image_path):
    # Perform damage detection using YOLOv5

    # Initialize results dictionary
    results = {'status': '', 'message': '', 'damage_detected': False}

    try:
        # Load the image for detection
        dataset = LoadImages(image_path, img_size=640)
        img_path, img, *_ = dataset[0]

        # Inference
        results = model(img)  # Perform object detection

        # Apply non-maximum suppression
        results = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)

        if results[0] is not None:
            # Damage detected
            results['status'] = 'success'
            results['message'] = 'Damage detected'
            results['damage_detected'] = True
        else:
            # No damage detected
            results['status'] = 'success'
            results['message'] = 'No damage detected'
            results['damage_detected'] = False

    except Exception as e:
        results['status'] = 'error'
        results['message'] = str(e)

    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform damage detection
        detection_results = detect_damage(filename)

        return jsonify(detection_results)

if __name__ == '__main__':
    app.run(debug=True)
