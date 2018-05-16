import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from bkgSubtraction import *

def find_center(image, rGuess, cGuess, radLow, radHigh, Nsamples=50, radSize=80):

  """
  def minFunct(center):
    c,r = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
    c -= center[0]
    r -= center[1]
    radMap = np.sqrt(c**2 + r**2)
    rmask,cmask = np.where((radMap>radLow) & (radMap<radHigh))
    values = np.reshape(image[rmask,cmask], (-1))
    print("pcent",center[0],center[1])
    
    return np.mean((values - np.mean(values))**2)

  centerEstimates = np.array([rGuess, cGuess])
  #centerR,centerC = optimize.fmin(minFunct, centerEstimates, ftol=0.0001)
  centerR,centerC = optimize.fmin_powell(minFunct, centerEstimates, ftol=0.0001)
  #center, ier = optimize.leastsq(minFunct, centerEstimates, full_output=True)

  return int(centerR), int(centerC)
  """

  def minFunct(center, *circle):
    diff = np.sum((circle - center)**2, axis=1)
    return np.mean(diff, axis=0)


  c,r = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
  c -= cGuess
  r -= rGuess
  radMap = np.sqrt(c**2 + r**2)
  radMid = (radLow + radHigh)/2.

  #itr = (radHigh - radLow)/Nsamples
  #for i in range(Nsamples):
  #rMask,cMask = np.where((radMap>radLow+i*itr) & (radMap<radLow+i*itr+radSize))
  rMask,cMask = np.where((radMap>radLow) & (radMap<radHigh))
  """
  plot = np.zeros_like(image)
  plot[rMask,cMask] = 1
  plt.imshow(plot)
  plt.show()
  """
  #rMask,cMask = np.where((temp>2) & (temp<4))
  mean = np.mean(image[rMask,cMask])
  std = np.std(image[rMask,cMask])

  rMask,cMask = np.where((image>mean-std/4) & (image<mean+std/4) &
                         (radMap>radMid-radSize) & (radMap<radMid+radSize))

  """
  plot = np.zeros_like(image)
  plot[rMask,cMask] = 1
  plt.imshow(plot)
  plt.show()
  """

  circle = np.concatenate((np.reshape(rMask,(-1,1)), np.reshape(cMask, (-1,1))), axis=1)
  centerEstimates = np.array([rGuess, cGuess])
  centerR,centerC = optimize.fmin(minFunct, centerEstimates, args=tuple(circle))

  return int(centerR), int(centerC)



def center_image(image, centerR, centerC, roi):
  roiRange = roi//2
  assert ((roiRange+centerR+1 < image.shape[0]) and (centerR-roiRange >= 0)),\
      "ERROR: row ROI {} is out of range for image of size {} and center row {}".format(
      roi+1, image.shape, centerR)
  assert ((roiRange+centerC+1 < image.shape[1]) and (centerC-roiRange >= 0)),\
      "ERROR: col ROI {} is out of range for image of size {} and center col {}".format(
      roi+1, image.shape, centerR)

  return image[centerR-roiRange:centerR+roiRange+1,centerC-roiRange:centerC+roiRange+1]


def centering(image, config):

  ###  center finding  ###
  if (config.centerC is not None) and (config.centerR is not None):
    image = center_image(image, config.centerR, config.centerC, config.roi)
    return image, config.centerR, config.centerC
  else:
    # naive readout noise subtraction
    image = readoutNoise_subtraction(image, False)
    centerR, centerC = find_center(image,
                          config.guessCenterR, config.guessCenterC,
                          config.centerRadLow, config.centerRadHigh)
    image = center_image(image, centerR, centerC, config.roi)
    return image, centerR, centerC


