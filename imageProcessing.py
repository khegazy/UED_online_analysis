from getImgInfo import *
from getImg import *
from bkgSubtraction import *
from centerFinding import *
from centerImg import *

def imageProcessing(name, config):
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

