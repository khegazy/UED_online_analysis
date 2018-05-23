import numpy as np
import os
import time
import glob
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

class CONFIG():
  def __init__(self):

    ###  file querying information  ###
    self.doQueryFolder = False
    self.queryFolder = "."
    self.queryBkgAddr = "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif"


    ###  loading results and files  ###
    # load saved results
    self.loadSavedResults = False
    self.loadSavedResultsFolder = "results"
    self.loadSavedResultsFileName = "allCHD_"

    # load saved files
    self.loadFolders = []
    #badRuns = ["019","020","024","025","016","017","018","006","009","096","111","118","010","138","149","035","039","077","007","086","008","091","026"]
    badRuns = ["096","111","118","010","138","149","035","039","077","007","086","008","091","026"]
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
    self.subFolder = "images-ANDOR1"
    self.fileExtention = "*.tif"


    ###  saving results  ###
    self.saveFolder = "results"
    self.saveFileName = "fullResults"
    self.saveQueryResults = False
    self.saveLoadedResults = True


    self.hotPixel = 7000
    self.bkgImgNames = [ 
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-001-024.4950_0001.tif",
          "/reg/ued/ana/scratch/CHD/20161212/LongScan2/run013/images-ANDOR1/ANDOR1_delayHigh-002-024.5850_0001.tif"]
    self.doCenterFind = [True, False]
    self.bkgNorms = [None, None]
    self.ROradLow = 0.9
    self.ROradHigh = 0.98
    self.normRadLow = 0.65
    self.normRadHigh = 0.95
    self.roi = 834
    self.guessCenterR = 530
    self.guessCenterC = 500
    self.centerRadLow = 148
    self.centerRadHigh = 152
    self.centerR = 550 #None
    self.centerC = 508 #None
    self.sumMin = 5.59e8
    self.sumMax = 5.72e8

    self.gMatrixFolder = "/reg/neh/home/khegazy/analysis/legendreFitMatrices/"
    self.Nrebin = 5
    self.Nlegendres = 6
    self.NradialBins = 50

    self.Qmax = 11.3
    self.normByAtomic = True
    self.atomicDiffractionFile = "/reg/neh/home5/khegazy/analysis/CHD/simulation/diffractionPattern/output/references/atomicScattering_CHD.dat"
    self.atomicDiffractionDataType = np.float64
    self.plotPrefix = ""


newNames = [
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

"""
while len(config.loadFolders):
  loadDict = config.loadFolders.pop(0)
  loadFiles = os.listdir(loadDict["folder"])  
  if loadDict["background"] in loadFiles:
    loadFiles.remove(loadDict["background"])

  ###  find the center for this run  ###
  loadConfig = CONFIG()
  loadConfig.centerR = loadDict.centerR
  loadConfig.centerC = loadDict.centerC

  if (loadConfig.centerC is None) or (loadConfig.centerR is None):
    for i in range(min(loadConfig.Ncenters, len(loadFiles))):
      ###  get image, remove hot pixels, get image norm  ###
      img = get_image(name, config.hotPixel)

      _, centerR, centerC = centering(img, loadConfig)
      centerRsum += centerR
      centerCsum += centerC
      centerSumCount += 1

    loadConfig.centerR = int(centerRsum/float(centerSumCount))
    loadConfig.centerC = int(centerCsum/float(centerSumCount))

  ###  get background image  ###
  loadbkg = get_image(loaddict["background"], config.hotpixel)

  imgSum    = np.zeros((config.roi, config.roi), np.float)
  imgNorms  = 0
  for i,name in enumerate(loadFiles):
    ###  get image, remove hot pixels, get image norm  ###
    img = get_image(name, config.hotPixel)

    ###  subtract background  ###
    img = img - loadBkg
    
    ###  centering image  ###
    img, _, _ = centering(img, loadConfig)

    ###  readout noise subtraction  ### 
    img = readoutNoise_subtraction(img, True, 
                rLow=config.ROradLow, rHigh=config.ROradHigh)

    ###  weighted sum of images  ###
    imgSum    += img
    imgNorms  += get_image_norm(img, loadConfig.normRadLow, loadConfig.normRadHigh))

  finalImg = imgSum/imgNorms
  finalImg.tofile(



  loadImages += finalImg
  loadNorms += 1.0


#####  centering images that can't use normal methods using average center  #####
for ind in reCenterImgs:
  img = center_image(bkgImages[ind], 
              centerR, centerC, config.roi)
  ## advanced readout noise subtraction 
  img = readoutNoise_subtraction(img, True, 
              rLow=config.ROradLow, rHigh=config.ROradHigh)

  ###  get image norm  ###
  if config.bkgNorms[i] is None:
    bkgImages[ind] = img[:,:]/get_image_norm(img, config.normRadLow, config.normRadHigh)
  else:
    bkgImages[ind] = img[:,:]/config.bkgNorms[ind]

"""








saveNorms = []


###  initialize file lists  ###
queryFiles = []
loadedFiles = []
loadFiles = []
for fld in config.loadFolders:
  folderName = fld["folder"] + "/" + config.fileExtention
  diffractionFiles = glob.glob(folderName)
  bkgImgFiles = [fld["background"]]*len(diffractionFiles)
  centerRs = [fld["centerR"]]*len(diffractionFiles)
  centerCs = [fld["centerC"]]*len(diffractionFiles)
  loadFiles = loadFiles + zip(diffractionFiles, bkgImgFiles, centerRs, centerCs)

while (len(loadFiles) == 0) and (len(queryFiles) == 0):
  if not config.doQueryFolder:
    print("ERROR: There are no files included in the load folders!")
    raise RuntimeError


###  initialize plots  ###
plt.ion()
Qrange = np.arange(config.NradialBins+1)*config.Qmax/(config.NradialBins)
fig,ax = plt.subplots(2, 1)

ax[1].set(xlabel="Time [ps]", ylabel=r'Q $[\AA^{-1}]$')
ax[0].set(xlabel=r'Q $[\AA^{-1}]$', ylabel="Legendre 0")

plot1d, = ax[0].plot(Qrange[:-1], np.zeros((config.NradialBins)), "b-")


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

  # initialize loadind variables with first new entry
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
    queryFiles = query_folder(config.queryFolder, config.fileExtention)
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

loadedFiles = []
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
    
    while len(queryFiles) == 0:
      print("INFO: Query folder is empty, waiting to check again")
      time.sleep(1)
      folderFiles = glob.glob(config.queryFolder + "/" + config.fileExtention)
      queryFiles = [fl for fl in folderFiles if fl not in loadedFiles]
    continue

  ###  get image information  ###
  info = get_image_info(name)

  ###  get image and remove hot pixels  ###
  img = get_image(name, config.hotPixel)
  saveNorms.append(np.sum(img))
  imgSum = np.sum(img)
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
  plot1d.set_ydata(legendreCoeffs[0,:])
  ax[0].set_ylim([0,legendreCoeffs[0,0]])


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

            


