import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':

  # Import command line arguments
  argP = argparse.ArgumentParser()
  argP.add_argument('--PolarFit', type = str, default = 'legendre',
      help = "Polar function to fit")
  argP.add_argument('--RadialFit', type = str, default = 'heavyside',
      help = "Radial function to fit")
  argP.add_argument('--NradBins', type = int, default = 50,
      help = "Number of radial bins after the fit")
  argP.add_argument('--Nlg', type = int, default = 6,
      help = "Number of legendres to fit to [0,Nlg)")
  argP.add_argument('--Nrows', type = int, default = -1,
      help = "Number of rows in the image")
  argP.add_argument('--Ncols', type = int, default = -1,
      help = "Number of columns in the image")
  argP.add_argument('--stdev', type = float, default = -1,
      help = "Standard deviation of the radial gaussian kernel")
  args = argP.parse_args()

  # Initialize variables
  kwargs = {}
  polarFit  = args.PolarFit
  radialFit = args.RadialFit
  NradBins  = args.NradBins
  Nlg       = args.Nlg
  Nrows     = args.Nrows
  Ncols     = args.Ncols

  if Nrows == -1 and Ncols != -1:
    Nrows = Ncols
  elif Nrows != -1 and Ncols == -1:
      Ncols = Nrows
  elif Nrows == -1 and Ncols == -1:
    raise RunTimeError("Must specify the number of rows and/or columns!!!")

  Npix      = Nrows*Ncols
  if args.stdev == -1:
    #stdev = Nrows/NradBins
    stdev = 1


  # row-major radii starting with rad=0 bin
  rowShift = 1 - Nrows%2
  colShift = 1 - Ncols%2
  x = np.array([np.arange(Ncols, dtype=np.float64) + 0.5*colShift - Ncols/2., ]*Nrows)
  y = np.array([np.arange(Nrows, dtype=np.float64) + 0.5*rowShift - Nrows/2., ]*Ncols).transpose()
  rad = np.sqrt(x**2 + y**2)
  rad_flat = np.reshape(rad, (-1, 1))

  #[Npix, NradBins]
  #k = np.array([np.arange(NradBins, dtype=np.float64) + 1., ]*Npix)
  #k *= Nrows/(NradBins + 1)
  #radBasis = np.exp(-(rad_flat[:,:] - k[:,:])**2/(2*stdev**2))
  
  #k = np.array([np.arange(NradBins, dtype=np.float64) + 1., ]*Npix)
  #k *= (Nrows/2)/(NradBins + 1)
  #radBasis = np.exp(-(rad_flat[:,:] - k[:,:])**2/(2*stdev**2))/(np.sqrt(2*np.pi)*stdev)

  ######################
  ###  Radial Basis  ###
  ######################
  radBasis = np.zeros((Npix, NradBins), dtype = float)
  if radialFit.lower() == 'heavyside': 
    k = rad_flat*(NradBins)/(Nrows/2)
    for i in range(Npix):
      for r in range(NradBins):
        if np.floor(k[i]) == r:
          radBasis[i,r] = 1.#/(r+1) #rad_flat[i]

  
  #####################
  ###  Polar Basis  ###
  #####################
  polarBasis = np.zeros((Nlg, Nrows, Ncols), dtype = float)
  if polarFit.lower() == 'legendre' :
    Ndiv = int(np.ceil(10000/Nrows))
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


  curFolder = os.path.dirname(os.path.realpath(__file__))
  print(curFolder)
  outputFile = open(curFolder + "/gMatrix_row-" + str(Nrows) + "_col-" \
      + str(Ncols) + "_Nrad-" + str(NradBins) \
      + "_Nlg-" + str(Nlg) + ".dat", "wb")

  g.tofile(outputFile)
  outputFile.close()

  #test = np.reshape(g[:,2], (Nrows, Ncols))
  #print(test)
  #plt.imshow(test)
  #plt.show()
  #print(g)

