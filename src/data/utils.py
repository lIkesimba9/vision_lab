import cv2
import numpy as np

def load_image(image_path, size=None) -> np.array:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if size is not None:
        image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.