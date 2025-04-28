from src.utils.config import VOC_CLASSES , MODEL_PATH
import torch
from torch import nn
from torchvision import models



class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False 
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)  

    def forward(self, x):
        x = self.base_model(x)
        x = torch.sigmoid(x)  
        return x


model = CustomResNet(num_classes=len(VOC_CLASSES))  
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) 
model.eval()
