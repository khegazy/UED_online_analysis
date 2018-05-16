import pickle as pl
import numpy as np
import os

def save_results(legDict, loadedFiles, legArray, folder, prefix=""):
  if not os.path.exists(folder):
    print("INFO: folder {} does not exist, now creating!!!".format(folder))
    os.makedir(folder)

  prefix = folder + "/" + prefix
  pl.dump(legDict, open(prefix + "averageLegCoeffDict.pl", "wb"))
  pl.dump(loadedFiles, open(prefix + "loadedFiles.pl", "wb"))
  legArray.tofile(open(prefix + "averageLegCoeffs.dat", "wb"))

  return

