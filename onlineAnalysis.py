import numpy as np
import os
import time
import glob
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from getImg import *
from makeLgMatrix import *
from bkgSubtraction import *
from rebinImg import *
from centering import *
from getImgNorm import *
from getImgInfo import *
from fitLegendres import *
from saveResults import *
from queryFolder import *

class CONFIG():
  def __init__(self):


    ###################################
    ###  file querying information  ###
    ###################################

    self.doQueryFolder  = True
    self.queryFolder    = "/reg/ued/ana/scratch/CHD/20161212/testScan"
    self.queryBkgAddr   = "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif"


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

      ind = fld.find("run")
      runNum = int(fld[ind+3:])
      if True: #(runNum > 80) and (runNum <= 180):

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

    # background images (will sum images in list)
    self.bkgImgNames = [ 
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif",
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-002-024.5850_0001.tif"]

    # hot pixels
    self.hotPixel = 7000

    # image region of interest
    self.roi = 834

    # readout noise subtraction range
    self.ROradLow   = 0.9
    self.ROradHigh  = 0.98

    # image normalization range
    self.normRadLow   = 0.65
    self.normRadHigh  = 0.95
    self.sumMin       = 5.59e8   # ignore all images with a total sum below this
    self.sumMax       = 5.72e8   # ignore all images with a total sum above this

    # center finding
    self.centerR        = 550    # use None to do center finding
    self.centerC        = 508    # use None to do center finding
    self.guessCenterR   = 530
    self.guessCenterC   = 500
    self.centerRadLow   = 148    # use this range to determine center
    self.centerRadHigh  = 152    # use this range to determine center

    # legendre fitting parameters
    self.Nlegendres     = 6      # will fit legendres [0, Nlegendres)
    self.NradialBins    = 50     # number of radial bins when fitting
    self.Nrebin         = 5      # rebin before fitting (makes fitting faster)
    self.gMatrixFolder  = "/reg/neh/home/khegazy/analysis/legendreFitMatrices/"


    #################################
    ###  diffraction information  ###
    #################################

    # q normalization
    self.QperPixel = 22.6/900 
    self.Qmax = self.QperPixel*self.roi/2

    # calculate sM(s) by normalizing with atomic diffraction
    self.normByAtomic = True
    self.atomicDiffractionFile = "/reg/neh/home5/khegazy/analysis/CHD/simulation/diffractionPattern/output/references/atomicScattering_CHD.dat"
    self.atomicDiffractionDataType = np.float64


    #############################
    ###  plotting parameters  ###
    #############################

    self.plotPrefix = ""
    self.plotFigSize = (14, 6)
    self.dpi = 80



config = CONFIG()




###################################
#####  get background images  #####
###################################

bkgImages = []
reCenterImgs = []
centerRsum = 0.
centerCsum = 0.
centerSumCount = 0.


saveNorms = []


###  initialize file lists  ###
loadedFiles = []
loadFiles = []
queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)
for fld in config.loadFolders:
  folderName = fld["folder"] + "/" + config.fileExtention
  diffractionFiles = glob.glob(folderName)
  bkgImgFiles = [fld["background"]]*len(diffractionFiles)
  centerRs = [fld["centerR"]]*len(diffractionFiles)
  centerCs = [fld["centerC"]]*len(diffractionFiles)
  loadFiles = loadFiles + zip(diffractionFiles, bkgImgFiles, centerRs, centerCs)

while (len(loadFiles) == 0) and (len(queryFiles) == 0):
  if not config.doQueryFolder:
    raise RuntimeError("ERROR: There are no files included in the load folders!")
  else:
    queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)
    while not len(queryFiles):
      print("There are no diffraction patterns under %s, will keep looking..." % config.queryFolder)
      time.sleep(10)
      queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)


###  initialize plots  ###
plt.ion()
Qrange = np.arange(config.NradialBins+1)*config.Qmax/(config.NradialBins)
fig,ax = plt.subplots(2, 4, figsize=config.plotFigSize, dpi=config.dpi)

ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)
ax[1,0].get_xaxis().set_visible(False)
ax[1,0].get_yaxis().set_visible(False)
ax[0,1].set(xlabel=r'Q $[\AA^{-1}]$', ylabel="")
ax[1,1].set(xlabel="Time", ylabel="Total Counts")
ax[0,2].set(xlabel="Time [ps]", ylabel=r'Q $[\AA^{-1}]$')
ax[1,2].set(xlabel="Time [ps]", ylabel="Legendre 0")
ax[0,3].set(xlabel="Time [ps]", ylabel=r'Q $[\AA^{-1}]$')
ax[1,3].set(xlabel="Time [ps]", ylabel="Legendre 2")

plotCurLeg, = ax[0,1].plot(Qrange[:-1], np.zeros((config.NradialBins)), "b-")
plotL0LO,   = ax[1,2].plot(Qrange[:-1], np.zeros((config.NradialBins)), "b-")
plotL2LO,   = ax[1,3].plot(Qrange[:-1], np.zeros((config.NradialBins)), "b-")


###  image variables  ###
aggregateImage = np.zeros((1024,1024), np.int32)
imageSums = []


###  initialize legendre variables  ###
legCoeffDict = {}
loadImgDict = {}
averageLegCoeffArray = np.zeros((config.Nlegendres,1,config.NradialBins))

initializeFiles = True
if config.loadSavedResults:
  legCoeffDict, loadedFiles, averageLegCoeffArray =\
      load_results(config.loadSavedResultsFolder, config.loadSavedResultsFileName)
  delays = np.sort(np.array(legCoeffDict.keys()))
  initializeFiles = False

  # initialize loading variables with first new entry
  while len(loadFiles):
    fileName,_,_,_ = loadFiles[0]
    if fileName in loadedFiles:
      del loadFiles[0]
    else:
      info = get_image_info(fileName)
      delays = np.array([info.stageDelay])
      loadImgDict[info.stageDelay] = (0, 0)
      break


while initializeFiles and\
    (initializeFiles or (len(loadFiles) is 0) or (len(queryFiles) is 0)):
  if len(loadFiles):
    fileName,_,_,_ = loadFiles[0]
    info = get_image_info(fileName)
    delays = np.array([info.stageDelay])
    loadImgDict[info.stageDelay] = (0, 0)
    initializeFiles = False
  elif len(queryFiles):
    info = get_image_info(queryFiles[0])
    delays = np.array([info.stageDelay])
    legCoeffDict[info.stageDelay] = (0, 0)
    initializeFiles = False
  elif config.doQueryFolder:
    queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)
  else:
    print("ERROR: Cannot run without loading files or querying folder!!!")
    sys.exit()

for i in np.arange(len(loadFiles)):
  fileName,_,_,_ = loadFiles[-1]
  imgSum = np.sum(get_image(fileName, config.hotPixel))
  if (imgSum < config.sumMin) or (imgSum > config.sumMax):
    del loadFiles[-1]
  else:
    break




###  retrieving atomic diffraction  ###
if config.normByAtomic:
  atomicDiffraction = np.fromfile(config.atomicDiffractionFile, 
      dtype=config.atomicDiffractionDataType)*1e20
  qGrid = (np.array(config.NradialBins, np.float) + 0.5)\
            *config.Qmax/config.NradialBins
  atomicNorm = 1./(atomicDiffraction*qGrid)

###  retrieve gMatrix for legendre fitting  ###
assert ((config.roi+1)%config.Nrebin == 0),\
    "ERROR: Cannot rebin an image with size [{}, {}] by {}, change roi!".format(
        config.roi+1, config.roi+1, config.Nrebin)
imgRebinSize = (config.roi+1)/config.Nrebin

gMatrixName = "gMatrix_pixels-" + str((config.roi+1)/config.Nrebin)\
              + "Nradii-" + str(config.NradialBins)\
              + "Nlegendre-" + str(config.Nlegendres) + ".dat"

if not os.path.isfile(config.gMatrixFolder + "/" + gMatrixName):
  make_legendre_gMatrix(config.NradialBins, config.Nlegendres,
                    imgRebinSize, config.gMatrixFolder + "/" + gMatrixName)

gMatrix = np.fromfile(config.gMatrixFolder + "/" + gMatrixName,
                        dtype=np.float)
gMatrix = np.reshape(gMatrix,
              (imgRebinSize**2, config.NradialBins*config.Nlegendres))

## invert g matrix using SVD decomposition
gInv = invert_matrix_SVD(gMatrix)

loadingImage = False
curBkgAddr = ""
loadConfig = CONFIG()
while (len(loadFiles) != 0) or config.doQueryFolder:
  if len(loadFiles):
    name, bkgAddr, loadConfig.centerR, loadConfig.centerC = loadFiles.pop(0)
    while name in loadedFiles:
      name, bkgAddr, loadConfig.centerR, loadConfig.centerC = loadFiles.pop(0)

    loadingImage = True
    centerConfig = loadConfig
    loadedFiles.append(name)

    # load background
    if curBkgAddr is not fld["background"]:
      curBkgAddr = fld["background"]
      if curBkgAddr is not None:
        bkgImg = get_image(fld["background"], config.hotPixel)

  elif len(queryFiles):
    name = queryFiles.pop(0)
    imgAddr = None
    loadingImage = False
    centerConfig = config
    loadedFiles.append(name)
 
    # load background
    if curBkgAddr is not config.queryBkgAddr:
      curBkgAddr = config.queryBkgAddr
      if curBkgAddr is not None:
        bkgImg = get_image(config.queryBkgAddr, config.hotPixel)

  else:
    ###  save current results  ###
    if config.saveQueryResults:
      save_results(legCoeffDict, loadedFiles, averageLegCoeffArray,
          config.saveFolder, config.saveFileName)

    ###  search query folder for new files  ###
    
    queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)
    while len(queryFiles) == 0:
      print("INFO: Query folder is empty, waiting to check again")
      time.sleep(1)
      queryFiles = query_folder(config.queryFolder, config.fileExtention, loadedFiles)
    continue

  print("Now looking at %s" % name)
  ###  get image information  ###
  info = get_image_info(name)

  ###  get image and remove hot pixels  ###
  imgOrig = get_image(name, config.hotPixel)
  aggregateImage += imgOrig
  img = copy.deepcopy(imgOrig)

  ###  check total scattering intensity  ###
  imgSum = np.sum(img)
  imageSums.append(imgSum)
  if (imgSum < config.sumMin) or (imgSum > config.sumMax):
    continue
  #plt.imshow(img)
  #plt.show()

  ###  subtract background images  ###
  if curBkgAddr is not None:
    img -= bkgImg #background_subtraction(img, bkgImg)
  #plt.imshow(img)
  #plt.show()

  ###  center image  ###
  img, centerR, centerC = centering(img, centerConfig)
  #print("centers", centerR, centerC)
  """
  if (config.centerR is not None) and (config.centerC is not None):
    centerR = config.centerR
    centerC = config.centerC
  else:
    centerR,centerC = find_center(img, 
                          config.guessCenterR, config.guessCenterC,
                          config.centerRadLow, config.centerRadHigh)

  img = center_image(img, centerR, centerC, config.roi)
  """
  #plt.imshow(img)
  #plt.show()

  ###  readout noise subtraction  ###
  img = readoutNoise_subtraction(img, True, 
              rLow=config.ROradLow, rHigh=config.ROradHigh)
  #plt.imshow(img)
  #plt.show()

  ###  image norm  ###
  imgNorm = 1#get_image_norm(img, config.normRadLow, config.normRadHigh)


  #####  update loaded images  #####
  if loadingImage:
    ind = np.searchsorted(delays, [info.stageDelay])[0]
    if np.any(np.abs(delays-info.stageDelay) < 0.005):
      delayInd = delays[ind]
      loadImg,norm = loadImgDict[delayInd] 
      loadImgDict[delayInd] = (loadImg + img, norm + imgNorm)
    else:
      delays = np.insert(delays, ind, info.stageDelay)
      loadImgDict[info.stageDelay] = (img, imgNorm)

    """
    legendreCoeffs = fit_legendres(img, config.Nrebin, config.Nlegendres,
                                     config.NradialBins, gInv=gInv)
    X,Y = np.meshgrid(np.array([0,1]), Qrange)
    fitTest = plt.figure()
    axTest = figTest.add_subplot(111)
    img = axTest.pcolor(X, Y, np.reshape(legendreCoeffs[0,:],(1,-1)).T, cmap=cm.jet)
        axLoad.set_ylim([0,config.Qmax])
        axLoad.set_xlim([timeDelay[0],timeDelay[-1]])
        figLoad.colorbar(img, ax=axLoad)
        img.set_clim(-1*maxRange, maxRange)
        figLoad.canvas.draw()
        figLoad.savefig("legednre" + str(i) + "_loadedFiles.png")


    """
    if len(loadFiles) == 0:
      averageLegCoeffArray = np.zeros((config.Nlegendres, 
                                       delays.shape[0], 
                                       config.NradialBins), np.float)
      for i,d in enumerate(delays):
        # fit legendres
        img,norm = loadImgDict[d]
        legendreCoeffs = fit_legendres(img, config.Nrebin, config.Nlegendres,
                                          config.NradialBins, gInv=gInv)
        if config.normByAtomic:
          legendreCoeffs *= atomicNorm 

        # record results
        legCoeffDict[d] = (legendreCoeffs, norm)
        averageLegCoeffArray[:,i,:] = legendreCoeffs/norm

      ###  save results  ###
      if config.saveLoadedResults:
        save_results(legCoeffDict, loadedFiles, averageLegCoeffArray,
          config.saveFolder, config.saveFileName)

      plt.hist(saveNorms, 30)
      plt.show()
      plt.savefig("normDist.png")


      ###  plot results of loaded files  ###
      timeDelay = (delays - delays[0])*1e-9/(3e8*1e-12)
      if timeDelay.shape[0] > 1:
        timeDelay = np.insert(timeDelay, -1, 2*timeDelay[-1]-timeDelay[-2])
      else:
        timeDelay = np.insert(timeDelay, -1, timeDelay[-1]+0.05)
      timeDelay = timeDelay[1:]
      X,Y = np.meshgrid(timeDelay, Qrange)

      for i in [0,2]:
        figLoad = plt.figure()
        axLoad = figLoad.add_subplot(111)

        subTZleg = averageLegCoeffArray[i] - np.mean(averageLegCoeffArray[i,:4,:], axis=0)
        shp = subTZleg.shape
        mn = np.mean(subTZleg[:,0.2*shp[1]:0.7*shp[1]], axis=(0,1))
        std = np.std(subTZleg[:,0.2*shp[1]:0.7*shp[1]], axis=(0,1))
        if mn > 0:
          maxRange = np.abs(mn - 3*std)
        else:
          maxRange = mn + 3*std
        #maxRange = 0.14

        #axLoad.pcolor(X, Y, averageLegCoeffArray[i,:,:].T, cmap=cm.RdBu)
        img = axLoad.pcolor(X, Y, subTZleg[1:,:].T, cmap=cm.jet)
        axLoad.set_ylim([0,config.Qmax])
        axLoad.set_xlim([timeDelay[0],timeDelay[-1]])
        figLoad.colorbar(img, ax=axLoad)
        img.set_clim(-1*maxRange, maxRange)
        figLoad.canvas.draw()
        figLoad.savefig("legednre" + str(i) + "_loadedFiles.png")

    continue


  #####  fit legendres  #####
  legendreCoeffs = fit_legendres(img, config.Nrebin, config.Nlegendres,
                                          config.NradialBins, gInv=gInv)
  if config.normByAtomic:
    legendreCoeffs *= atomicNorm 

  #####  update time domain legendres  #####
  ind = np.searchsorted(delays, [info.stageDelay])[0]
  print(delays)
  print(ind, info.stageDelay)
  print(np.any((delays-info.stageDelay) == 0))
  if np.any((delays-info.stageDelay) == 0):
    delayInd = delays[ind]
    coeffs,norm = legCoeffDict[delayInd] 
    updatedCoeffs = coeffs + legendreCoeffs
    legCoeffDict[delayInd] = (updatedCoeffs, norm + imgNorm)
    averageLegCoeffArray[:,ind,:] = updatedCoeffs[:,:]/(norm + imgNorm)
  else:
    delays = np.insert(delays, ind, info.stageDelay)
    averageLegCoeffArray = np.insert(averageLegCoeffArray, ind, 
                                      legendreCoeffs[:,:], axis=1)
    legCoeffDict[info.stageDelay] = (legendreCoeffs, imgNorm)

  #####  plot time domain legendre fits  #####

  ###  diffraction patterns  ###
  ax[0,0].imshow(imgOrig)
  ax[1,0].imshow(aggregateImage)

  plotCurLeg.set_ydata(legendreCoeffs[0,:])
  #ax[0,1].set_ylim([0,legendreCoeffs[0,0]])

  ax[1,1].plot(np.arange(len(imageSums)), imageSums)

  ###  time dependent plots  ###

  timeDelay = (delays - delays[0])*1e-2/(3e8*1e-12)
  if timeDelay.shape[0] > 1:
    timeDelay = np.insert(timeDelay, -1, 2*timeDelay[-1]-timeDelay[-2])
  else:
    timeDelay = np.insert(timeDelay, -1, timeDelay[-1]+0.05)
  X,Y = np.meshgrid(timeDelay, Qrange)
  #axLegAll.pcolor(Qrange, timeDelay, averageLegCoeffArray[0,:,:], cmap=cm.RdBu)
  # aggregate legendre 0 plot
  ax[0,2].pcolor(X, Y, averageLegCoeffArray[0,:,:].T, cmap=cm.RdBu)
  ax[0,2].set_ylim([0,config.Qmax])
  ax[0,2].set_xlim([timeDelay[0],timeDelay[-1]])

  lineOut = np.sum(averageLegCoeffArray[0,:,10:20], axis=1)
  plotL0LO.set_data(timeDelay[:-1], lineOut)
  ax[1,2].set_ylim([min(lineOut),max(lineOut)])
  ax[1,2].set_xlim([timeDelay[0],timeDelay[-1]])
  #plotL0LO.set_xdata(timeDelay[:-1])
  #plotL0LO.set_ydata(lineOut)

  # aggregate legendre 2 plot
  ax[0,3].pcolor(X, Y, averageLegCoeffArray[2,:,:].T, cmap=cm.RdBu)
  ax[0,3].set_ylim([0,config.Qmax])
  ax[0,3].set_xlim([timeDelay[0],timeDelay[-1]])

  lineOut = np.sum(averageLegCoeffArray[2,:,10:20], axis=1)
  plotL2LO.set_data(timeDelay[:-1], lineOut)
  ax[1,3].set_ylim([min(lineOut),max(lineOut)])
  ax[1,3].set_xlim([timeDelay[0],timeDelay[-1]])
  #plotL2LO.set_xdata(timeDelay[:-1])
  #plotL2LO.set_ydata(lineOut)

  fig.canvas.draw()



################################
#####  plot final results  #####
################################

finalFig = plt.figure()
ax = finalFig.add_subplot(111)

timeDelay = (delays - delays[0])*1e-2/(3e8*1e-12)
if timeDelay.shape[0] > 1:
  timeDelay = np.insert(timeDelay, -1, 2*timeDelay[-1]-timeDelay[-2])
else:
  timeDelay = np.insert(timeDelay, -1, timeDelay[-1]+0.05)
X,Y = np.meshgrid(timeDelay, Qrange)

for i in range(config.Nlegendres):
  ax.pcolor(X, Y, averageLegCoeffArray[i,:,:].T, cmap=cm.RdBu)
  finalFig.savefig(config.plotPrefix + "Legendre" + str(i) + ".png")

            


