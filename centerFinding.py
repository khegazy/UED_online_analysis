import nympy as np
import scipy as sc


def minFunct(c):


def find_center(image, rGuess, cGuess, radLow, radHigh):

  def minFunct(center):
    c,r = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
    c -= center[0]
    r -= center[1]
    radMap = np.sqrt(c**2 + r**2)
    mask = np.where(radMap>radLow and radMap<radHigh)
    values = np.reshape(image[mask[0],mask[1]], (-1))
    
    return values - np.mean(values)

  centerEstimates = rGuess,cGuess
  center, ier = optimize.leastsq(minFunct, centerEsitmates)

  return center


