import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import requests, io

class_names = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vehicles over 3.5 tons","Right-of-way at next intersection","Priority road","Yield","Stop","No vehicles","Vehicles over 3.5 tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Snow/ice warning","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing for vehicles over 3.5 tons"]

class GTSRBModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.fc_layers = nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,128),nn.ReLU(),nn.Linear(128,43))
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = GTSRBModelCNN()
model.load_state_dict(torch.load("gtsrb_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

url = input("Enter image URL: ")
img = Image.open(io.BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content)).convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.inference_mode():
    probs = torch.softmax(model(tensor), dim=1)
    confidence, idx = probs.max(dim=1)

print(f"Predicted : {class_names[idx.item()]}")
