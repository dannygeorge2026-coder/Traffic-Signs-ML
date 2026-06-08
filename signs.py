import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB

transform = transforms.Compose([
    transforms.Resize((32, 32)),   # resize all images to 32x32
    transforms.ToTensor()
])

train_data = GTSRB(root="data", split="train", transform=transform, download=True)
test_data = GTSRB(root="data", split="test", transform=transform, download=True)



class_names = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Snow/ice warning",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 tons"
]



from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 64
# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)


import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# Import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

class GTSRBModelCNN(nn.Module):
    def __init__(self, output_shape: int = 43):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_1 = GTSRBModelCNN(output_shape=43)
model_1.to(device)

from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Setup loss function and optimizer for model_1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 15

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_1.train()
        # 1. Forward pass
        y_pred = model_1(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_1.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model_1(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Save the model
torch.save(model_1.state_dict(), "gtsrb_model.pth")
print("Saved PyTorch Model State to gtsrb_model.pth")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_1.parameters()).device))
import requests
from PIL import Image
import io

def load_and_preprocess_image_from_url(image_url: str, transform) -> torch.Tensor:
    """Loads an image from a URL, applies transformations, and returns a preprocessed tensor."""
    # Download the image with a User-Agent header to mimic a browser request
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(image_url, headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes
    image_bytes = io.BytesIO(response.content)

    # Open the image
    img = Image.open(image_bytes).convert("RGB") # Ensure image is RGB

    # Apply the same transformations as training data
    transformed_img = transform(img)

    # Add a batch dimension (model expects [batch_size, channels, height, width])
    return transformed_img.unsqueeze(0)
                                         
def predict_image_class(model: torch.nn.Module, image_tensor: torch.Tensor, class_names: list) -> str:
    """Makes a prediction on a preprocessed image tensor using the given model."""
    model.eval() # Set model to evaluation mode
    image_tensor = image_tensor.to(next(model.parameters()).device)
    with torch.inference_mode():
        # Make prediction
        logits = model(image_tensor)
        pred_prob = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1).item()

    # Get the predicted class name
    return class_names[pred_label]
                                    
    
# Example Google image URL (replace with any valid traffic sign image URL)
# This is an example of a 'Stop' sign.
# You might need to find your own image URLs for different signs.
# The previous URLs caused errors due to direct download restrictions or not found issues.
# Let's try a stable public image from a GitHub repository.
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCccY5rEOozow2tguGelmB4RfBM_38j2CuQ&s"

# Load and preprocess the image using the 'transform' defined earlier
preprocessed_image_tensor = load_and_preprocess_image_from_url(image_url, transform)

# Make a prediction
predicted_class_name = predict_image_class(model_1, preprocessed_image_tensor, class_names)

print(f"The predicted traffic sign is: {predicted_class_name}")

# Display the image as well
plt.imshow(preprocessed_image_tensor.squeeze().permute(1, 2, 0))
plt.title(f"Predicted: {predicted_class_name}")
plt.axis("off")
plt.show()