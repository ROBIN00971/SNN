# To run this Python code, you need the following libraries:
# pip install Flask Pillow torch torchvision snntorch flask-cors

import numpy as np
import io
import torch
import torch.nn as nn
import snntorch as snn
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from PIL import Image

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- 1. Define the Spiking Neural Network Model Architecture ---
# This class must be identical to the one in your trainer.py file
class SpikingNet(nn.Module):
    def __init__(self):
        super(SpikingNet, self).__init__()
        
        # Define the spiking neuron model: Leaky Integrate-and-Fire (LIF)
        beta = 0.95 # Neuron decay rate

        # Layer 1: Fully-connected layer followed by a LIF neuron
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=beta)

        # Layer 2: Fully-connected layer followed by a LIF neuron
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize membrane potentials and outputs for each layer
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []

        for step in range(25): # Assuming num_steps = 25 as in trainer.py
            cur1 = self.fc1(x[step].view(x[step].size(0), -1))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), mem2

# --- 2. Load the Trained Model ---
# This will be run once when the server starts
model_path = 'snn_mnist_model.pth'
try:
    snn_model = SpikingNet()
    snn_model.load_state_dict(torch.load(model_path))
    snn_model.eval() # Set the model to evaluation mode
    print("SNN model loaded successfully.")
except FileNotFoundError:
    snn_model = None
    print(f"Error: Model file '{model_path}' not found. Please train the model first by running trainer.py.")
except Exception as e:
    snn_model = None
    print(f"An error occurred while loading the model: {str(e)}")


# --- 3. Define Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# --- 4. API Endpoint to Evaluate Image ---
@app.route('/evaluate-image', methods=['POST'])
def evaluate_image():
    if snn_model is None:
        return jsonify({'error': 'Model not loaded. Please train the model and restart the server.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            pil_image = Image.open(io.BytesIO(file.read())).convert('L')
            
            # Pre-process the image for the model
            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(1) # Add a batch dimension
            
            # --- Generate the spike train for the model ---
            num_steps = 25
            data_spikes = snn.spikegen.rate(image_tensor, num_steps=num_steps)

            # --- Forward pass through the SNN model to get a prediction ---
            with torch.no_grad():
                spk_out, mem_out = snn_model(data_spikes)
            
            # The model predicts the digit with the highest total spike count
            _, predicted_class = torch.sum(spk_out, dim=0).max(1)
            predicted_digit = predicted_class.item()
            
            message = f'Image processed successfully! The SNN model predicts the digit is: {predicted_digit}'

            return jsonify({
                'message': message,
                'status': 'success',
                'prediction': predicted_digit
            })
            
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
