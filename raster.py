# https://gis.stackexchange.com/questions/306861/split-geotiff-into-multiple-cells-with-rasterio
from shapely import geometry
from rasterio.mask import mask
import rasterio
import collections
import numpy
import csv
import pickle
import logging

samplesList = []

# Takes a Rasterio dataset and splits it into squares of dimensions squareDim * squareDim
def splitImageIntoCells(img, filename, squareDim, saveSamplesToDisc):
    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[0] // squareDim
    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh ):
        y = hc * squareDim
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            geom = getTileGeom(img.transform, x, y, squareDim)
            getCellFromGeom(img, geom, filename, count, saveSamplesToDisc)
            print ("hc: {}".format(hc))
            print ("wc: {}".format(wc))
            count = count + 1

# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1],
                        corner2[0], corner2[1])

# Crop the dataset using the generated box and write it out as a GeoTIFF
def getCellFromGeom(img, geom, filename, count, saveSamplesToDisc):
    crop, cropTransform = mask(img, [geom], crop=True)
    if saveSamplesToDisc:
        writeImageAsGeoTIFF(crop,
                        cropTransform,
                        img.meta,
                        img.crs,
                        filename+"gndSmp_"+str(count))
    unique, counts = numpy.unique(crop[5], return_counts=True) # unique - array with Id of classes in sampel, counts - amount of class in sampel
    classDictionery = dict(zip(unique, counts))
    classId = max(classDictionery, key=classDictionery.get) # gets the class with gratest amount of pixels in sample
    tempArray = ["gndSmp_"+str(count), classId, classDictionery.get(classId)/1024 ] #[name , classID, % of classID pixels]
    samplesList.append( tempArray )

# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": crs})
    with rasterio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)

# input interface: 
#                   rasterPath - path to main raster
#                   destinationPath - path were samples will be saved [ all samples will be named "gndSmp_X"]
#                   fileName - path + name of List of all samples [ name, classId, % ]
#                   saveSamplesToDisc - bool if small samples of raster should be saved
def sampleriseRaster( rasterPath, destinationPath, fileName, saveSamplesToDisc ):
    src = rasterio.open( rasterPath )
    logging.info('SampleriseRaster start!' )
    splitImageIntoCells( src, destinationPath, 32, saveSamplesToDisc)
    logging.info('SampleriseRaster stop!' )
    logging.info('List %s size is: %s', fileName, len(samplesList) )
    f = open( fileName + ".pkl","wb")
    pickle.dump(samplesList,f)
    f.close()
    return samplesList
