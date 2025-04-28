import torchvision.transforms as transforms
from src.utils.config import VOC_CLASSES
from src.utils.load_model import model
from PIL import Image
import numpy as np
from io import BytesIO
import torch




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def Predict(image):
    """
    Predict the class of the image using the pre-trained model.

    Args:
        image (bytes): The input image in bytes to classify.

    Returns:
        dict: A dictionary containing the predicted label(s) and confidence score(s).
    """
    img = Image.open(BytesIO(image)).convert("RGB")
    
    img_tensor =  transform(img).unsqueeze(0)
    
    output = model(img_tensor)
    
    probabilities = output
    
    threshold = 0.5
    
    predicted_indices = torch.where(probabilities[0] > threshold)[0].tolist()
    
    predicted_labels =  [VOC_CLASSES[i] for i in predicted_indices ]  
    
    predicted_confidences = [probabilities[0][i].item() for i in predicted_indices]
    
    return {"classes": predicted_labels, "confidence": predicted_confidences} 
    
   
