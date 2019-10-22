# satelliteImageClassifier
CNN network (VGG16 architecture) for classifying different land types on Sentinel images

README STILL underconstruction!

The aim of this project is comperison of neural network performance regarding quality of input data, evaluation of lerning process. An convolutional neural network, for classifying land types such as: wather, forest, city and agricultural lands, has been implemented. The result of classification is displayed as a map. As an input data Sentinel-2B satelite images ware used. Firstly they ware preclassyfied by human for training and testing pourposes. VGG architecture was used as an core of convolutional neural network.

## Basics
### Input
As an input you will need hi-resolution image. In this case we use 5 band image (RGB + infrared + NDVI) plus one band of pixel classes.

### Classifiaction
Network classifies 4 different classes with ID from 0 to 3. ID's are kept in 6th band (each pixel has its own ID). Network takes 32x32 pixels samples and classifaies it.

### Training&Testing
1/5 of data provided in dataConversion.prepBatches() will be usead for testing pourposes. It is due to fact that we create 5 batches of data and one is for testing.
10% of training data will be saved for validation

### Output
All logs will be saved to file

## Files
All workflow is triggered from main.py. It gives abillity to ran it on different input data sets.
