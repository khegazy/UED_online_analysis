import numpy as np

def rebin_image(image, rebin):
  assert (image.shape[0]%rebin == 0),\
      "ERROR: Number of rows {} cannot be rebinned by {}!".format(
          image.shape[0], rebin)
  assert (image.shape[1]%rebin == 0),\
      "ERROR: Number of cols {} cannot be rebinned by {}!".format(
          image.shape[1], rebin)

  rRows = int(image.shape[0]/rebin)
  cRows = int(image.shape[1]/rebin)
  rImg = np.zeros((rRows, cRows))
  for ir in range(rRows):
    for ic in range(cRows):
      rImg[ir,ic] = np.mean(
                      image[ir*rebin:(ir+1)*rebin,ic*rebin:(ic+1)*rebin])

  return rImg
