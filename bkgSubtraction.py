import numpy as np


def readoutNoise_subtraction(image, isCentered, rLow=990, cLow=10, rHigh=1010, cHigh=30, numAvg=50):
  if isCentered:
    if (rLow > 1.) or (rHigh >1.):
      print("ERROR: rLow and rHigh are percentages (>0 and <1) of the radius to use!!!")
      raise RuntimeError

    c,r = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
    c -= image.shape[1]//2
    r -= image.shape[0]//2
    rad = np.sqrt(c**2 + r**2)/(image.shape[0]/2)
    radMask = np.where((rad>rLow) & (rad<rHigh))
    noise = np.mean(image[radMask])

  else:
    assert ((rLow>=0) and (rHigh<image.shape[0])),\
        "Error: Readout noise row range ({}, {}) must be within [0, {}]!".format(
            rLow, rHigh, image.shape[0])
    assert ((cLow>=0) and (cHigh<image.shape[1])),\
        "Error: Readout noise col range ({}, {}) must be within [0, {}]!".format(
            cLow, cHigh, image.shape[1])

    pixels = np.reshape(image[rLow:rHigh,cLow:cHigh], (-1))
    pads = max(pixels.shape[0] - numAvg, 0)//2
    noise = np.mean(pixels[pads:pixels.shape[0]-1-pads])

  return image - noise



def background_subtraction(image, bkgImages=[]):
  return image - np.sum(np.array(bkgImages), axis=0)

