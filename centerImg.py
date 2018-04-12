import numpy as np

def center_image(image, centerR, centerC, roi):
  roiRange = roi//2
  assert ((roiRange+centerR < image.shape[0]) and (centerR-roiRange >= 0)),\
      "ERROR: row ROI {} is out of range for image of size {} and center row {}".format(
      roi, image.shape, centerR)
  assert ((roiRange+centerC < image.shape[1]) and (centerC-roiRange >= 0)),\
      "ERROR: col ROI {} is out of range for image of size {} and center col {}".format(
      roi, image.shape, centerR)

  return image[centerR-roiRange:centerR+roiRange+1,centerC-roiRange:centerC+roiRange+1]
