import glob

def query_folder(folder, fileExtention, loadedFiles):
  folderFiles = glob.glob(folder + "/" + fileExtention)
  return [fl for fl in folderFiles if fl not in loadedFiles]
