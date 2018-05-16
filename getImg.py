from PIL import Image
import numpy as np


def get_image(img_name, hotPixel, windowSize=10, dataType=np.float):
  inpImg = Image.open(img_name)
  img = np.array(inpImg, dtype=dataType)

  ###########################
  ###  Remove Hot Pixels  ###
  ###########################
  indR, indC = np.where(img > hotPixel)
  shift = windowSize//2
  rSize = img.shape[0]
  cSize = img.shape[1]
  for r,c in zip(indR,indC):
    newPixel = np.median(
                img[max(0,r-shift):min(r+shift,rSize-1),
                    max(0,c-shift):min(c+shift,cSize-1)])
    img[r,c] = newPixel

  #meshR,meshC = np.meshgrid(np.arange(
  #normMask = img

  return img
