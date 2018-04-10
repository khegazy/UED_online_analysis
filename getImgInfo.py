import numpy as np


class imgINFO:
  imgNum
  stageDelay



def get_image_info(name):
  info = imgINFO

  ind1 = name.find("delayHigh")
  ind1 = name.find("-", ind1)
  ind2 = name.find("-", ind1+1)
  print("imgnum", name[ind1:ind2])
  info.imgNumber = int(name[ind1:ind2])

  ind1 = name.find("_", ind2)
  print("delay", name[ind2:ind1])
  info.stageDelay = float(name[ind2:ind1])

  return info

