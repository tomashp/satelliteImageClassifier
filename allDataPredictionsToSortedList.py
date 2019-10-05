import pickle
import os
import re
lista = "allDataPredictions.pkl"
path = "D:/FullResolution1/allDataPredictions.pkl"

dictlist = []

with open(path + lista, mode='rb') as file:
    # note the encoding type is 'latin1'
    batch = pickle.load(file, encoding='latin1')

for key, value in batch.items():
    sampleID = re.findall('\d+', key )
    sampleID = int( sampleID[0] )
    temp = [ sampleID , int(value)]
    dictlist.append(temp)

dictlist.sort(key=lambda x: x[0])

with open('your_file.txt', 'w') as f:
    for item in dictlist:
        f.write("%s\n" % item)
f.close()