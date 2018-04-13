import numpy as np
from rebinImg import *


def invert_matrix_SVD(matrix):
  U,s,V = np.linalg.svd(matrix, full_matrices=False)
  sInv = np.reshape(1./s, (-1,1))
  sInvUtrans = np.multiply(sInv, np.transpose(U))
  return np.dot(np.transpose(V), sInvUtrans)


def fit_legendres(image, Nrebin, Nlegendres, NradialBins,
                  gMatrix=None, gInv=None):

  assert ((gMatrix is not None) or (gInv is not None)),\
      "ERROR: Must supply gMatrix (if image has nans) or gInv!"

  #####  Fit Legendres  #####
  imgRebin = rebin_image(image, Nrebin)

  ## invert g matrix
  if gInv is None:
    gInv = invert_matrix_SVD(gMatrix)

  ## flatten image
  imgFlat = np.reshape(imgRebin, (-1))

  ## fit legendres
  lgC = np.dot(gInv, imgFlat)
  return np.reshape(lgC, (Nlegendres, NradialBins))

