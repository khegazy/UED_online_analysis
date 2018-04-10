import numpy as np

def center_image(image, centerR, centerC, roiR, roiC):
  rRange = roiR//2
  cRange = roiC//2
  if (rRange+centerR => image.shape[0]) or (centerR-rRange < 0):
    print("ERROR: row ROI {} is out of range for image of size {} and center row {}".format(
      roiR, image.shape, centerR)
    raise RuntimeError
  if (cRange+centerC => image.shape[1]) or (centerC-cRange < 0):
    print("ERROR: col ROI {} is out of range for image of size {} and center col {}".format(
      roiR, image.shape, centerR)
    raise RuntimeError

  return image[centerR-rRange:centerR+cRange,centerC-cRange:centerC+cRange]
