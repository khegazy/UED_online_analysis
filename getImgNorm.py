import numpy as np

def get_image_norm(image, radLow, radHigh):

  assert ((radLow>=0.) and (radLow<=1.)),\
    "ERROR: Radii argument radLow for image norm must be percentage 0>x>1!"
  assert ((radHigh>=0.) and (radHigh<=1.)),\
    "ERROR: Radii argument radHigh for image norm must be percentage 0>x>1!"

  c,r = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
  c -= image.shape[1]//2
  r -= image.shape[0]//2
  rad = np.sqrt(c**2 + r**2)/(image.shape[0]/2)
  rMask,cMask = np.where((rad>radLow) & (rad<radHigh))
  return np.mean(image[rMask,cMask])


