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
  # Use int to avoid float precision error. 
  # nanometer precision to work for all future fine adjustments
  info.stageDelay = int(float(name[ind2+1:ind1])*1e6)

  return info

