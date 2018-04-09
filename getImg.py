from PIL import Image
import numpy as np


def get_image(img_name, dataType):
  img = Image.open(img_name)
  return np.array(img, dtype=dataType)
