from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import base64
from io import BytesIO

app = Flask(__name__)

# Load the ML model
model = timm.create_model(model_name="resnet50", pretrained=True, num_classes=5, in_chans=1)
model.load_state_dict(torch.load('models/resnet_best_model.pth', map_location=torch.device('cpu'))) 
model.eval()


def predict_image(img):
    # Preprocess the image
    img = img.resize((64, 64))  
    img = img.convert("L")  
    img = np.array(img) / 255.0 
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make predictions
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)

    # Interpret predictions
    labels_map = {0: 'tops', 1: 'bottoms', 2: 'bags', 3: 'shoes', 4: 'others'}
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

    pred = labels_map[predicted_class_index]

    return pred

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        image = Image.open(file)

        input_image = image_to_base64(image)
        
        prediction = predict_image(image)
        
        return render_template('index.html', prediction=prediction, image=input_image)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)