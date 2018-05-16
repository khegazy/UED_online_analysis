import pickle as pl
import numpy as np
import os
import sys
import glob

def save_results(legDict, loadedFiles, legArray, folder, prefix=""):
  if not os.path.exists(folder):
    print("INFO: folder {} does not exist, now creating!!!".format(folder))
    os.makedir(folder)

  prefix = folder + "/" + prefix
  pl.dump(legDict, open(prefix + "averageLegCoeffDict.pl", "wb"))
  pl.dump(loadedFiles, open(prefix + "loadedFiles.pl", "wb"))

  shape = list(legArray.shape)
  flatArray = np.reshape(legArray, (1,-1))
  arrayName = prefix + "averageLegCoeffs_shape"
  os.remove(arrayName + "*dat")
  for sh in shape:
    arrayName += "-" + str(sh)
  arrayName += ".dat"
  flatArray.tofile(open(arrayName, "wb"))

  return


def load_results(folder, prefix):
  legDict, loadedFiles, legArray = None, None, None
  path = folder + "/" + prefix
  files = glob.glob(path + "*")
  print(files)
  try:
    print("INFO: Searching for " + path + "averageLegCoeffDict.pl!")
    if (path + "averageLegCoeffDict.pl") in files:
      legDict = pl.load(open(path + "averageLegCoeffDict.pl", "rb"))
    else:
      raise ImportError
    print("INFO: Searching for " + path + "loadedFiles.pl!")
    if (path + "loadedFiles.pl") in files:
      loadedFiles = pl.load(open(path + "loadedFiles.pl", "rb"))
    else:
      raise ImportError
    print("INFO: Searching for " + path + "averageLegCoeffs.dat!")
    if any((path + "averageLegCoeffs") in fl for fl in files):
      arrayName = glob.glob(path + "averageLegCoeffs*.dat")[0]
      legArray = np.fromfile(arrayName, dtype=float)

      ind = arrayName.find("shape-") + 5
      end = arrayName.find(".dat")
      shape = []
      while ind is not end:
        indNext = arrayName.find("-", ind+1)
        if indNext is -1:
          indNext = end
        print(ind, indNext)
        shape.append(int(arrayName[ind+1:indNext]))
        ind = indNext

      legArray = np.reshape(legArray, shape)

    else:
      raise ImportError
  except ImportError:
    print("ERROR: Cannot find file above, please check path!!!\n\n")
    sys.exit()

  return legDict, loadedFiles, legArray


