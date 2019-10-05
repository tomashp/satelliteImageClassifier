import raster               # cuts tiff to small 32x32 tiffs, create List of all samples [ name, classId, % of pixels of the class]
import dataConversion       # base on sampleList creates 4 classes and gets the best quality data, then prepers dictBatches for input to CNN
import training             # process ( normalize, one-hot ) the batches and Trains CNN
import evalTrainings        # evaluates network performance on one of the batches prepered for training ( the one that was not used for training )
import evaluating           # evaluates network performance on full data ( all samples from raster )
import processAllDataForValidation  # all data sampels needs to be prepered for validation ( normalized, one-hot )
import logging
import pickle
import os
# RUN SCRIPT FROM WHERE YOU WANT TO SAVE FILES [anaconda: cd /d d:\Docs\Java] https://stackoverflow.com/questions/11065421/command-prompt-wont-change-directory-to-another-drive
# CHOOSE the option
processName = "HalfResolution"
#processName = "FullResolution"

logging.basicConfig(filename= processName + '.log',level=logging.INFO,format='%(asctime)s %(message)s')
logging.info('Process of %s is starting', processName )




#defines
if processName is "FullResolution":
    mainRasterPath = 'C:/Users/PlochaTo/Documents/TP/PG/MGR/rasterDoPodzialu.tif'
    samplesPaths = "D:/cutData32/"
    dataset_folder_path = 'D:/FullResolution1/'
else:
    mainRasterPath = 'C:/Users/PlochaTo/Documents/TP/PG/MGR/rasterDoPodzialuSmooth.tif'
    samplesPaths = "D:/cutData32HalfResolution/"
    dataset_folder_path = 'D:/HalfResolution1/'

# change saveSamplesToDisc = True if you are cuting raster for first time.
saveSamplesToDisc = False    

save_model_path = dataset_folder_path + 'model/image_classification'
allDataPreProcessedPath = dataset_folder_path + 'preprocessedAllDataBatch/'

# we need to make sure that all directoris are ready
if not os.path.exists(samplesPaths):
    os.makedirs(samplesPaths)
    
if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)

if not os.path.exists(dataset_folder_path + 'model/'):
    os.makedirs(dataset_folder_path + 'model/')

if not os.path.exists(allDataPreProcessedPath):
    os.makedirs(allDataPreProcessedPath)

logging.info('Main raster: %s', mainRasterPath )
logging.info('Samples from main raster %s. Will be save to disc: %s', samplesPaths, saveSamplesToDisc )
logging.info('Dataset folder path %s', dataset_folder_path )
logging.info('Model will be saved %s', save_model_path )
logging.info('All data from raster will be kept in %s', allDataPreProcessedPath )

# make 32x32 samples saves them at samplesPaths 
# and create List of all samples [ name, classId, % of pixels of the class] and saves it to dataset_folder_path
#samplesList = raster.sampleriseRaster(mainRasterPath, samplesPaths, dataset_folder_path + processName + "SamplesList", saveSamplesToDisc)
#if you skip line before uncoment below
with open(dataset_folder_path + processName + "SamplesList.pkl", mode='rb') as file:
   samplesList = pickle.load(file, encoding='latin1')

# prepers input for CNN
# dataConversion.prepBatches( samplesList, samplesPaths, 7000, dataset_folder_path + processName + "TrainingBatch")       # for prepered set
dataConversion.prepBatches( samplesList, samplesPaths, 61168, allDataPreProcessedPath + processName + "AllDataBatch")   # for All data set

# trains the network
#training.run(dataset_folder_path, save_model_path, processName + "TrainingBatch")
# test data evaluating
#evalTrainings.test_model( save_model_path, dataset_folder_path )
# evaluate network on all data
#processAllDataForValidation.preprocess_and_save_data(allDataPreProcessedPath  + processName )
evaluating.test_model( save_model_path, allDataPreProcessedPath ) # allDataPreProcessedPath must be path to folder, all data '.p' will be used  during evaluation 