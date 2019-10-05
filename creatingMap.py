# https://gis.stackexchange.com/questions/306861/split-geotiff-into-multiple-cells-with-rasterio
from shapely import geometry
from rasterio.mask import mask
import rasterio
import collections
import numpy
import csv
import pickle
import logging
import cv2
import re
import numpy as np

classifiedList = []
maskList = []

# Takes a Rasterio dataset and splits it into squares of dimensions squareDim * squareDim
def createClassifiedMap(img, squareDim, transform):
    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[2] // squareDim
    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh ):
        y = hc * squareDim
        if count == 20:
                break
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            classifieGeom(img, count , x, y, squareDim, transform)
            print ("hc: {}".format(hc))
            print ("wc: {}".format(wc))
            count = count + 1
            if count == 20:
                break


# Crop the dataset using the generated box and write it out as a GeoTIFF https://datacarpentry.org/image-processing/04-drawing-bitwise/
def classifieGeom(img, count, x, y, squareDim, transform):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    tempMask = np.zeros(shape = img.shape, dtype = "float32")
    # Draw a white, filled rectangle on the mask image
    color = classifiedList[count][1] # assigns class ID as a color
    cv2.rectangle(img = tempMask, 
        pt1 = (int(corner1[0]), int(corner1[1])), pt2 = (int(corner2[0]), int(corner2[1])), 
        color = (color,color,color), 
        thickness = int(-1))
    
    # Apply the mask and display the result
    img = cv2.bitwise_and(src1 = img, src2 = tempMask)

# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": crs})
    with rasterio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)

def readList( path ):
    with open(path , mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    for key, value in batch.items():
        sampleID = re.findall('\d+', key ) # get only number from image names ( we will use the same csount in this module )
        sampleID = int( sampleID[0] )
        temp = [ sampleID , int(value)]
        classifiedList.append(temp)

    classifiedList.sort(key=lambda x: x[0]) # we will get list of count and classes assigned in order same as our count goes in this module

# def createMasks():
#     classZeroMask = np.zeros(shape = src.shape, dtype = "uint8")
#     maskList[0] = classZeroMask.fill(0)
#     classOneMask = np.zeros(shape = src.shape, dtype = "uint8")
#     maskList[1] = classOneMask.fill(1)
#     classTwoMask = np.zeros(shape = src.shape, dtype = "uint8")
#     maskList[2] = classTwoMask.fill(2)
#     classThreeMask = np.zeros(shape = src.shape, dtype = "uint8")
#     maskList[3] = classThreeMask.fill(3)
    
rasterPath = 'C:/Users/PlochaTo/Documents/TP/PG/MGR/rasterDoPodzialu.tif'
destinationPath = 'D:/'
listDestination = "D:/FullResolution1/allDataPredictions.pkl"
#rasterPath = 'C:/Users/PlochaTo/Documents/TP/PG/MGR/rasterDoPodzialuSmooth.tif'
src = rasterio.open( rasterPath )

array = src.read()
# Create the basic black image 
# createMasks()

readList(listDestination)
createClassifiedMap(array, 32, src.transform)
# with rasterio.open(destinationPath + "test.tif", "w", driver='GTiff', height=array.shape[0], width=array.shape[1], crs=src.crs, count=1, dtype=array.dtype , transform=src.transform) as dest:
#     dest.write(array)
with rasterio.open(destinationPath + "fullResolutionClassified.tif", "w", **src.meta) as dest:
        dest.write(array)

print( " OH yess ")

