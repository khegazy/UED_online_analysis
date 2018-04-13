import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def make_legendre_gMatrix(NradBins, Nlg, size, fileName, 
          polarFit="legendre", radialFit="heavyside", stdev=-1):

  Npix = size*size
  if stdev == -1:
    #stdev = size/NradBins
    stdev = 1


  # row-major radii starting with rad=0 bin
  rowShift = 1 - size%2
  colShift = 1 - size%2
  x = np.array([np.arange(size, dtype=np.float64) + 0.5*colShift - size/2., ]*size)
  y = np.array([np.arange(size, dtype=np.float64) + 0.5*rowShift - size/2., ]*size).transpose()
  rad = np.sqrt(x**2 + y**2)
  rad_flat = np.reshape(rad, (-1, 1))

  #[Npix, NradBins]
  #k = np.array([np.arange(NradBins, dtype=np.float64) + 1., ]*Npix)
  #k *= size/(NradBins + 1)
  #radBasis = np.exp(-(rad_flat[:,:] - k[:,:])**2/(2*stdev**2))
  
  #k = np.array([np.arange(NradBins, dtype=np.float64) + 1., ]*Npix)
  #k *= (size/2)/(NradBins + 1)
  #radBasis = np.exp(-(rad_flat[:,:] - k[:,:])**2/(2*stdev**2))/(np.sqrt(2*np.pi)*stdev)

  ######################
  ###  Radial Basis  ###
  ######################
  radBasis = np.zeros((Npix, NradBins), dtype = float)
  if radialFit.lower() == 'heavyside': 
    k = rad_flat*(NradBins)/(size/2)
    for i in range(Npix):
      for r in range(NradBins):
        if np.floor(k[i]) == r:
          radBasis[i,r] = 1.#/(r+1) #rad_flat[i]

  
  #####################
  ###  Polar Basis  ###
  #####################
  polarBasis = np.zeros((Nlg, size, size), dtype = float)
  if polarFit.lower() == 'legendre' :
    Ndiv = int(np.ceil(10000/size))
    Ndiv += 1 - Ndiv%2
    intgrt = np.arange(Ndiv) - int(Ndiv/2)
    delta = 1.0/float(Ndiv)
    for ilg in range(Nlg):
      for ir in intgrt:
        for ic in intgrt:
          coeff = np.zeros(ilg + 1)
          coeff[-1] = 1
          rad_temp = np.sqrt((x + ic*delta)**2 + (y + ir*delta)**2)
          rad_temp[rad_temp == 0] = 1e-5
          cosTh = (x + ic*delta)/rad_temp 
          polarBasis[ilg,:,:] += np.polynomial.legendre.legval(cosTh, coeff)
    polarBasis /= float(Ndiv**2)
    polarBasis_Flat = np.reshape(polarBasis, (Nlg, -1, 1))


  #[Npix, Nlg*NradBins]
  g = np.zeros((Npix, Nlg*NradBins), dtype=np.float64)
  for ilg in range(Nlg) :
    coeff = np.zeros(ilg + 1)
    coeff[-1] = 1
    g[:, ilg*NradBins:(ilg+1)*NradBins] = radBasis*polarBasis_Flat[ilg,:,:] 


  outputFile = open(fileName, "wb")

  g.tofile(outputFile)
  outputFile.close()

  #test = np.reshape(g[:,2], (size, size))
  #print(test)
  #plt.imshow(test)
  #plt.show()
  #print(g)

