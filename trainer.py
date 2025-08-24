# To run this code, you need to install the following libraries:
# pip install torch torchvision snntorch tqdm

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm # For a progress bar during training

# --- 1. Set Up Device and Hyperparameters ---
# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Network hyperparameters
num_inputs = 28*28  # MNIST image size
num_hidden = 1000   # Number of neurons in the hidden layer
num_outputs = 10    # Number of output classes (0-9)
num_steps = 25      # Number of time steps for the spike simulation

# Training hyperparameters
batch_size = 128
epochs = 5
learning_rate = 1e-3

# --- 2. Load and Prepare the Data ---
# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Load the MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

print("Dataset loaded successfully.")

# --- 3. Define the Spiking Neural Network Model ---
# This is a simple feedforward network with two spiking layers.
class SpikingNet(nn.Module):
    def __init__(self):
        super(SpikingNet, self).__init__()

        # Define the spiking neuron model: Leaky Integrate-and-Fire (LIF)
        beta = 0.95 # Neuron decay rate

        # Layer 1: Fully-connected layer followed by a LIF neuron
        # NOTE: We no longer pass the surrogate_function here.
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)

        # Layer 2: Fully-connected layer followed by a LIF neuron
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
    
    # Define the surrogate gradient function outside of the init method
    spike_grad = surrogate.fast_sigmoid()

    def forward(self, x):
        # Initialize membrane potentials and outputs for each layer
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Create empty lists to store outputs from the spiking layers
        spk_rec = []
        mem_rec = []

        # Loop over each time step
        for step in range(num_steps):
            # Input to the first layer at the current time step
            cur1 = self.fc1(x[step].view(x[step].size(0), -1))
            # The surrogate function is now passed during the forward pass
            spk1, mem1 = self.lif1(cur1, mem1)

            # Input to the second layer
            cur2 = self.fc2(spk1)
            # The surrogate function is now passed during the forward pass
            spk2, mem2 = self.lif2(cur2, mem2)

            # Record the outputs
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

# Instantiate the network, loss function, and optimizer
net = SpikingNet().to(device)
loss_fn = SF.ce_rate_loss() # Cross-entropy loss for rate-coded SNNs
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- 4. Training Loop ---
print("Starting training...")
for epoch in range(epochs):
    # Set the network to training mode
    net.train()
    total_loss = 0
    with tqdm(train_loader, unit="batch") as t:
        t.set_description(f"Epoch {epoch+1}/{epochs}")
        for data, targets in t:
            data = data.to(device)
            targets = targets.to(device)

            # Generate spike train from the input image (rate-based encoding)
            data_spikes = spikegen.rate(data, num_steps=num_steps)

            # Reset the network's state for each batch
            utils.reset(net)
            
            # Forward pass through the network
            spk_out, _ = net(data_spikes)
            
            # Calculate loss
            loss = loss_fn(spk_out, targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Corrected line: Check if t.n is non-zero before dividing.
            if t.n > 0:
                t.set_postfix(loss=total_loss/t.n)

    print(f"Epoch {epoch+1} finished. Average Loss: {total_loss/len(train_loader):.4f}")

# --- 5. Test the Model's Accuracy ---
print("\nTesting model accuracy...")
with torch.no_grad():
    net.eval()
    correct_count = 0
    total_count = 0

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Generate spike train for test data
        data_spikes = spikegen.rate(data, num_steps=num_steps)
        utils.reset(net)

        # Forward pass
        spk_out, _ = net(data_spikes)
        
        # Calculate the final prediction from the output spikes
        # We take the neuron with the highest firing rate as the prediction
        _, predicted_class = torch.sum(spk_out, dim=0).max(1)
        
        total_count += targets.size(0)
        correct_count += (predicted_class == targets).sum().item()

    accuracy = 100 * correct_count / total_count
    print(f"Model Accuracy: {accuracy:.2f}%")

# --- 6. Save the Trained Model ---
# The state_dict contains all the learned weights
torch.save(net.state_dict(), 'snn_mnist_model.pth')
print("Model saved to snn_mnist_model.pth")
