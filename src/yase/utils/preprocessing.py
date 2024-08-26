import torch
import cv2

def preprocess(input_data, task="segmentation", size=(224, 224)):
    if isinstance(input_data, str):
        image = cv2.imread(input_data)
    else:
        image = input_data
    
    if task == "segmentation":
        image = cv2.resize(image, size)
    elif task == "depth":
        image = cv2.resize(image, size)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image / 255.0
