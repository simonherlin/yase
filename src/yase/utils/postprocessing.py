import torch
import cv2
import numpy as np

def postprocess(output, task="segmentation"):
    if task == "segmentation":
        output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        return output
    elif task == "depth":
        output = output.squeeze().cpu().numpy()
        return cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
