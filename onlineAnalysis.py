import numpy as np
import os.path
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from getImg import *
from makeLgMatrix import *
from bkgSubtraction import *
from rebinImg import *
from centerFinding import *
from centerImg import *
from getImgNorm import *
from getImgInfo import *

class CONFIG():
  def __init__(self):
    self.hotPixel = 1e5
    self.bkgImgNames = [ 
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif",
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-002-024.5850_0001.tif"]
    self.doCenterFind = [True, False]
    self.bkgNorms = [None, None]
    self.ROradLow = 0.9
    self.ROradHigh = 0.98
    self.normRadLow = 0.7
    self.normRadHigh = 0.9
    self.roi = 804
    self.guessCenterR = 530
    self.guessCenterC = 500
    self.centerRadLow = 148
    self.centerRadHigh = 152
    self.centerR = None
    self.centerC = None

    self.gMatrixFolder = "/reg/neh/home/khegazy/analysis/legendreFitMatrices/"
    self.Nrebin = 5
    self.Nlegendres = 6
    self.NradialBins = 5

    self.Qmax = 11.3

imgNames = [
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-002-024.5850_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-003-024.6300_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-004-024.4950_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-005-024.4800_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-006-024.6000_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-007-024.5400_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-008-024.3700_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-009-024.6150_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-010-024.4500_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-011-024.5700_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-012-024.4650_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-013-024.5250_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-014-024.5100_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-015-024.7950_0001.tif",
  "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-016-024.5550_0001.tif"]


config = CONFIG()

###################################
#####  get background images  #####
###################################
bkgImages = []
reCenterImgs = []
centerRsum = 0.
centerCsum = 0.
centerSumCount = 0.
print(config.hotPixel)
print(config.bkgImgNames)
for i,name in enumerate(config.bkgImgNames):
  ###  get image, remove hot pixels, get image norm  ###
  img = get_image(name, config.hotPixel)

  ###  naive readout noise subtraction  ###
  img = readoutNoise_subtraction(img, False) 
              #rLow=config.rLow, cLow=config.cLow,
              #rHigh=config.rHigh, cHigh=config.cHigh)

  ###  center finding  ###
  if config.doCenterFind[i]:
    if (config.centerC is not None) and (config.centerR is not None):
      img = center_image(img, config.centerR, config.centerC, config.roi)
      ## advanced readout noise subtraction 
      img = redoutNoise_subtraction(img, True, 
                rLow=config.ROradLow, rHigh=config.ROradHigh)
    else:
        centerR,centerC = find_center(img, 
                              config.guessCenterR, config.guessCenterC,
                              config.centerRadLow, config.centerRadHigh)
        centerRsum += centerR
        centerCsum += centerC
        centerSumCount += 1
        img = center_image(img, centerR, centerC, config.roi)
        ## advanced readout noise subtraction 
        img = readoutNoise_subtraction(img, True, 
                    rLow=config.ROradLow, rHigh=config.ROradHigh)

        ###  get image norm  ###
        if config.bkgNorms[i] is None:
          bkgImages.append(img[:,:]/get_image_norm(img, config.normRadLow, config.normRadHigh)) 
        else:
          bkgImages.append(img[:,:]/config.bkgNorms[i])
  else:
    reCenterImgs.append(i)
    bkgImages.append(img[:,:])



#####  centering images that can't use normal methods using average center  #####
for ind in reCenterImgs:
  img = center_image(bkgImages[ind], 
              int(centerRsum/centerSumCount), int(centerCsum/centerSumCount),
              config.roi)
  ## advanced readout noise subtraction 
  img = readoutNoise_subtraction(img, True, 
              rLow=config.ROradLow, rHigh=config.ROradHigh)

  ###  get image norm  ###
  if config.bkgNorms[i] is None:
    bkgImages[ind] = img[:,:]/get_image_norm(img, config.normRadLow, config.normRadHigh)
  else:
    bkgImages[ind] = img[:,:]/config.bkgNorms[ind]

plt.ion()
Qrange = np.arange(config.NradialBins+1)*config.Qmax/(config.NradialBins)
fig,ax = plt.subplots(2, 1)

ax[1].set(xlabel="Time [ps]", ylabel=r'Q $[\AA^{-1}]$')
ax[0].set(xlabel=r'Q $[\AA^{-1}]$', ylabel="Legendre 0")

plot1d, = ax[0].plot(Qrange[:-1], np.zeros((config.NradialBins)), "b-")
averageLegCoeffDict = {}
averageLegCoeffArray = np.zeros((config.Nlegendres,1,config.NradialBins))

info = get_image_info(imgNames[0])
delays = np.array([info.stageDelay], dtype=np.float)
averageLegCoeffDict[info.stageDelay] = (0, 0)

for name in imgNames:
  print(name)
  ###  get image information  ###
  info = get_image_info(name)

  ###  get image and remove hot pixels  ###
  img = get_image(name, config.hotPixel)
  #plt.imshow(img)
  #plt.show()

  ###  center image  ###
  if (config.centerR is not None) and (config.centerC is not None):
    centerR = config.centerR
    centerC = config.centerC
  else:
    centerR,centerC = find_center(img, 
                          config.guessCenterR, config.guessCenterC,
                          config.centerRadLow, config.centerRadHigh)

  print("center",centerR, centerC)
  img = center_image(img, centerR, centerC, config.roi)
  #plt.imshow(img)
  #plt.show()

  ###  readout noise subtraction  ###
  img = readoutNoise_subtraction(img, True, 
              rLow=config.ROradLow, rHigh=config.ROradHigh)
  #plt.imshow(img)
  #plt.show()

  ###  subtract background images  ###
  img = background_subtraction(img, bkgImages)
  #plt.imshow(img)
  #plt.show()

  #####  Fit Legendres  #####
  imgRebin = rebin_image(img, config.Nrebin)  
  ## get G matrix
  gMatrixName = "gMatrix_pixels-" + str(imgRebin.shape[0])\
                + "Nradii-" + str(config.NradialBins)\
                + "Nlegendre-" + str(config.Nlegendres) + ".dat"

  if not os.path.isfile(config.gMatrixFolder + "/" + gMatrixName):
    make_legendre_gMatrix(config.NradialBins, config.Nlegendres, 
                      imgRebin.shape[0], config.gMatrixFolder + "/" + gMatrixName)

  gMatrix = np.fromfile(config.gMatrixFolder + "/" + gMatrixName,
                          dtype=np.float)
  gMatrix = np.reshape(gMatrix, 
                (imgRebin.shape[0]**2, config.NradialBins*config.Nlegendres))

  ## invert g matrix
  U,s,V = np.linalg.svd(gMatrix, full_matrices=False)
  sInv = np.reshape(1./s, (-1,1))
  sInvUtrans = np.multiply(sInv, np.transpose(U))
  gInv = np.dot(np.transpose(V), sInvUtrans)

  ## flatten image
  imgFlat = np.reshape(imgRebin, (-1))

  ## fit legendres
  lgC = np.dot(gInv, imgFlat)
  legendreCoefficients = np.reshape(lgC, (config.Nlegendres, config.NradialBins))
  #plt.imshow(legendreCoefficients)
  #plt.show()

  #####  update time domain legendres  #####
  ind = np.searchsorted(delays, [info.stageDelay])[0]
  if np.any(np.abs(delays-info.stageDelay) < 0.005):
    delayInd = delays[ind]
    coeffs,Navg = averageLegCoeffDict[delayInd] 
    updatedCoeffs = (coeffs*Navg + legendreCoefficients)/(Navg + 1)
    averageLegCoeffDict[delayInd] = (updatedCoeffs, Navg + 1)
    averageLegCoeffArray[:,ind,:] = np.reshape(updatedCoeffs, 
                                          (config.Nlegendres, config.NradialBins))
  else:
    delays = np.insert(delays, ind, info.stageDelay)
    averageLegCoeffArray = np.insert(averageLegCoeffArray, ind, 
              np.reshape(legendreCoefficients, (config.Nlegendres, config.NradialBins)),
              axis=1)
    averageLegCoeffDict[info.stageDelay] = (legendreCoefficients, 1)

  #####  plot time domain legendre fits  #####
  plot1d.set_ydata(legendreCoefficients[0,:])
  ax[0].set_ylim([0,legendreCoefficients[0,0]])


  timeDelay = (delays - delays[0])*1e-2/(3e8*1e-12)
  if timeDelay.shape[0] > 1:
    timeDelay = np.insert(timeDelay, -1, 2*timeDelay[-1]-timeDelay[-2])
  else:
    timeDelay = np.insert(timeDelay, -1, timeDelay[-1]+0.05)
  X,Y = np.meshgrid(timeDelay, Qrange)
  print(X,Y)
  #axLegAll.pcolor(Qrange, timeDelay, averageLegCoeffArray[0,:,:], cmap=cm.RdBu)
  ax[1].pcolor(X, Y, averageLegCoeffArray[0,:,:].T, cmap=cm.RdBu)
  ax[1].set_ylim([0,config.Qmax])
  ax[1].set_xlim([timeDelay[0],timeDelay[-1]])
  fig.canvas.draw()

            


