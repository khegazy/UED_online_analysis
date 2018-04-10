import numpy as np
import matplotlib.pyplot as plt
from getImg import *
from makeLgMatrix import *
from bkgSubtraction import *

class config {
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
}

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


###################################
#####  get background images  #####
###################################
bkgImages = []
centerRsum = 0.
centerCsum = 0.
centerSumCount = 0.
for i,name in enumerate(bkgImgNames):
  #####  get image, remove hot pixels, get image norm  #####
  img = get_image(name, config.hotPixel)

  #####  naive readout noise subtraction  #####
  img = redoutNoise_subtraction(img, False, 
              rLow=config.rLow, cLow=config.cLow,
              rHigh=config.rHigh, cHigh=config.cHigh)

  #####  center finding  #####
  if config.doCenterFind[i]:
    centerRsum += centerR
    centerCsum += centerC
    centerSumCount += 1
    img = center_image(img, centerR, centerC, config.roiR, config.roiC)
    ## advanced readout noise subtraction 
    img = redoutNoise_subtraction(img, True, 
                rLow=config.ROradLow, rHigh=config.ROradHigh)

  #####  get image norm  #####
  if config.bkgNorms[i] is None:
    bkgImages.append(img/get_imageNorm(img, config.normRadLow, config.normRadHigh)) 
  else:
    bkgImages.append(img/config.bkgNorms[i])



print("hi")
for name in imgNames:
  print(name)

  #####  get image, remove hot pixels, get image norm  #####
  img,imgNorm = get_image(name, hotPixel)

  #####  subtrac background images  #####
  img = background_subtraction(img, bkgImages)
  plt.imshow(img)
  plt.show()



