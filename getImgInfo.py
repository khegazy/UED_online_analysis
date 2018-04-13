import numpy as np


class imgINFO():
  def __inin__(self):
    self.imgNumber = -1
    self.stageDelay = -1.


def get_image_info(name):
  info = imgINFO

  ind1 = name.find("delayHigh")
  ind1 = name.find("-", ind1)
  ind2 = name.find("-", ind1+1)
  info.imgNumber = int(name[ind1+1:ind2])

  ind1 = name.find("_", ind2)
  info.stageDelay = float(name[ind2+1:ind1])

  return info

