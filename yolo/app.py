from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the YOLOv8 models
models = []
models.append(YOLO("yolov8m-pose.pt"))
models.append(YOLO("yolov8m-seg.pt"))
models.append(YOLO("yolov8m.pt"))

def process_image_with_model(image, model):
    # Perform inference with YOLOv8
    results = model(image)
    
    # Get the first result from results
    result = results[0]
    
    # Render the detection results on the original image
    annotated_frame = result.plot()

    return annotated_frame

def process_image(image_data):
    # Decode the base64 image data
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = np.array(image)

    # Process image with both models
    annotated_frames = [process_image_with_model(image, model) for model in models]
    
    # Convert the images back to base64
    buffers = [cv2.imencode('.jpg', frame)[1] for frame in annotated_frames]
    encoded_images = [base64.b64encode(buffer).decode('utf-8') for buffer in buffers]
    
    return encoded_images

@socketio.on('image')
def handle_image(data):
    image_data = data['image']
    processed_images = process_image(image_data)
    
    return_value = {}
    for i in range(len(processed_images)):
        return_value[f"image{i+1}"] = processed_images[i]
    
    emit('processed_image', return_value)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
