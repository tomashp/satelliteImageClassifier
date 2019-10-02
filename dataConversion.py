from rasterio.mask import mask
import rasterio
import collections
import numpy
import csv
import pickle
import os
import csv
import random
import logging

classList = [ [],[],[],[] ]
listBatch0 = []
listBatch1 = []
listBatch2 = []
listBatch3 = []
listBatch4 = []

# Create list for each batch 
def createListForBatches( samplesNumber ):
    # for each batch we add one sample from each class, then we move to next batch
    sampleID = 0
    while ( sampleID < samplesNumber ): # we want 5 batches witch all samples used
        # ListBatch0
        for classID in range(0 ,4):
            if ( sampleID < len(classList[classID]) ):
                    listBatch0.append( classList[classID][sampleID] )
        sampleID += 1

        # ListBatch1
        for classID in range(0 ,4):
            if ( sampleID < len(classList[classID]) ):
                    listBatch1.append( classList[classID][sampleID] )
        sampleID += 1

        # ListBatch2
        for classID in range(0 ,4):
            if ( sampleID < len(classList[classID]) ):
                    listBatch2.append( classList[classID][sampleID] )
        sampleID += 1

        # ListBatch3
        for classID in range(0 ,4):
            if ( sampleID < len(classList[classID]) ):
                    listBatch3.append( classList[classID][sampleID] )
        sampleID += 1

        # ListBatch4
        for classID in range(0 ,4):
            if ( sampleID < len(classList[classID]) ):
                    listBatch4.append( classList[classID][sampleID] )
        sampleID += 1

#shuffle data on batchList
def shuffleData( ):
    random.shuffle(listBatch0)
    random.shuffle(listBatch0)
    random.shuffle(listBatch0)
    random.shuffle(listBatch1)
    random.shuffle(listBatch1)
    random.shuffle(listBatch1)
    random.shuffle(listBatch2)
    random.shuffle(listBatch2)
    random.shuffle(listBatch2)
    random.shuffle(listBatch3)
    random.shuffle(listBatch3)
    random.shuffle(listBatch3)
    random.shuffle(listBatch4)
    random.shuffle(listBatch4)
    random.shuffle(listBatch4)

#create array base on batchList
def createNumpyBatches( batchFileList, samplesPaths, x ):
    tempData = numpy.empty(shape=(1,32,32,5))
    tempLabel = []
    tempName = []
    y = 0
    for idx in batchFileList:
        fullPath = samplesPaths + idx[0] + '.tif'
        sample = rasterio.open(fullPath)
        array = sample.read()
        array = numpy.delete(array,5,0)                 #deleting class Id - won't be used during training
        array = array.transpose(1, 2, 0)                #(num_channel, width, height) -> (width, height, num_channel)
        array = numpy.expand_dims(array, axis = 0)
        tempData = numpy.vstack([tempData,array])
        tempName.append(idx[0])
        tempLabel.append(idx[1])
        y+=1
        print( "Lista: ", x, " sampel: " , y)

    tempData = numpy.delete(tempData, 0, 0) # need to delete first row beacuse is phantom: tempData = numpy.empty(shape=(1,32,32,5))
    batchDict = { "data" : tempData, "labels" : tempLabel, "name" : tempName }
    return batchDict

# take third element for sort - percentage od pixels of biggest class
def takeThird(elem):
    return elem[2]

# data for CNN preparation - interface
#                sampleList - list of all data samples
#                samplesDestPath - path to 32x32 data samples
#                numberOfSampels - number of samples whith will be used from sampleList
#                batchName - path + name of batches
def prepBatches( sampleList, samplesDestPath, numberOfSampels, batchName ):
    logging.info('Preparing data batches for  %s samples', numberOfSampels )
    for sample in sampleList:       #creates Lists for each class
        if sample[1] == 0:
            classList[0].append(sample)
        elif sample[1] == 1:
            classList[1].append(sample)
        elif sample[1] == 2:
            classList[2].append(sample)
        elif sample[1] == 3:
            classList[3].append(sample)

    for aList in classList:     #Sorts lists -> best samples at the begining
        aList.sort(key=takeThird, reverse=True)

    createListForBatches( numberOfSampels ) # Create list for each batch 

    shuffleData() # shuffles each list - data can not be sorted

    dictBatch0 = createNumpyBatches(listBatch0, samplesDestPath , 0)
    logging.info('Batch: %s0 size is: %s', batchName, len(listBatch0) )
    f = open( batchName + "0.pkl","wb")
    pickle.dump(dictBatch0,f)
    f.close()

    dictBatch1 = createNumpyBatches(listBatch1, samplesDestPath , 1)
    logging.info('Batch: %s1 size is: %s', batchName, len(listBatch1) )
    f = open( batchName + "1.pkl","wb")
    pickle.dump(dictBatch1,f)
    f.close()

    dictBatch2 = createNumpyBatches(listBatch2, samplesDestPath , 2)
    logging.info('Batch: %s2 size is: %s', batchName, len(listBatch2) )
    f = open( batchName + "2.pkl","wb")
    pickle.dump(dictBatch2,f)
    f.close()

    dictBatch3 = createNumpyBatches(listBatch3, samplesDestPath , 3)
    logging.info('Batch: %s3 size is: %s', batchName, len(listBatch3) )
    f = open( batchName + "3.pkl","wb")
    pickle.dump(dictBatch3,f)
    f.close()

    dictBatch4 = createNumpyBatches(listBatch4, samplesDestPath , 4)
    logging.info('Batch: %s4 size is: %s', batchName, len(listBatch4) )
    f = open( batchName + "4.pkl","wb")
    pickle.dump(dictBatch4,f)
    f.close()


