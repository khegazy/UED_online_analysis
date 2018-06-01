from onlineAnalysis import onlineAnalysis
import numpy as np
import os
import time
import glob
import copy
import matplotlib.pyplot as plt


class CONFIG():
  def __init__(self):


    ###################################
    ###  file querying information  ###
    ###################################

    self.doQueryFolder  = True
    self.queryBkgAddr   = None
    self.queryFolder    = "testData"
    self.fileExtention  = ".tif"


    ########################
    ###  saving results  ###
    ########################

    self.saveQueryResults   = False
    self.saveLoadedResults  = False
    self.saveFolder         = "results"
    self.saveFileName       = "fullResults"


    ##################################
    ###  loading results and data  ###
    ##################################

    # load previous data
    self.loadFolders    = []       # do not load anything if empty
    self.fileExtention  = ".tif"
    """
    badRuns = ["096","111","1i18","010","138","149","035","039","077","007","086","008","091","026"]
    for fld in glob.glob("/reg/ued/ana/scratch/CHD/20161212/LongScan2/run*"):
      skip = False
      for i in badRuns:
        if "run"+i in fld:
          skip = True
          break
      if skip:
        continue

        print(fld)
        self.loadFolders.append({
            "folder": fld + "/images-ANDOR1", 
            "background": None,
            #"background": "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif",
            "centerR" : 550,
            "centerC" : 508})
    """

    # load saved results (saved by this analysis)
    self.loadSavedResults           = False
    self.loadSavedResultsFolder     = "results"
    self.loadSavedResultsFileName   = "allCHD_"


    #############################
    ###  analysis parameters  ###
    #############################

    # hot pixels
    self.hotPixel = 7000

    # image region of interest
    self.roi = 835               # if not odd, will round up to odd number

    # readout noise subtraction range
    self.ROradLow   = 0.9        # [0.85, 0.95]
    self.ROradHigh  = 0.98       # [0.95, 0.99]
    
    # image normalization range
    self.normRadLow   = 0.65     # [0.5, 0.75]
    self.normRadHigh  = 0.95     # [0.9, 0.99]
    self.sumMin       = 5.59e8   # ignore all images with a total sum below this [5e8, 5.7e8]
    self.sumMax       = 5.72e8   # ignore all images with a total sum above this [5.5e5, 6e8]

    # center finding
    self.centerR        = 550    # use None to do center finding  
    self.centerC        = 508    # use None to do center finding  
    self.guessCenterR   = 530    # [530, 570]
    self.guessCenterC   = 500    # [480, 520]
    self.centerRadLow   = 148    # use this range to determine center
    self.centerRadHigh  = 152    # use this range to determine center

    # legendre fitting parameters
    self.Nlegendres     = 6      # will fit legendres [0, Nlegendres)
    self.NradialBins    = 50     # number of radial bins when fitting
    self.Nrebin         = 5      # rebin before fitting, must be factor of roi (faster fitting)
    self.gMatrixFolder  = "./legendreFitMatrices/"
    
    
    #################################
    ###  diffraction information  ###
    #################################

    # q normalization
    self.QperPixel = 22.6/900 
    self.Qmax = self.QperPixel*self.roi/2

    # calculate sM(s) by normalizing with atomic diffraction
    self.normByAtomic = False
    self.atomicDiffractionFile = "/reg/neh/home5/khegazy/analysis/CHD/simulation/diffractionPattern/output/references/atomicScattering_CHD.dat"
    self.atomicDiffractionDataType = np.float64


    #############################
    ###  plotting parameters  ###
    #############################

    self.plotPrefix = ""
    self.plotFigSize = (14, 6)
    self.dpi = 80


config = CONFIG()


onlineAnalysis(config)    
