import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def load_label_names():
    return ['other', 'water', 'fores', 'city']

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 4))
    
    for idx, val in enumerate(x):
        encoded[idx][int(val)] = 1
    
    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, name, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels, name), open(filename, 'wb'))


def preprocess_and_save_data(dataset_folder_path):
    n_batches = 5

    for batch_i in range(0, n_batches): # batch0 is a test batch
        # load the test dataset
        with open(dataset_folder_path + 'AllDataBatch' + str(batch_i) + ".pkl", mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # preprocess the testing data
        features = batch['data']
        labels = batch['labels']
        name = batch['name']

        # Preprocess and Save all testing data
        _preprocess_and_save(normalize, one_hot_encode,
                            np.array(features), np.array(labels), name,
                            dataset_folder_path + 'preprocess_AllData' + str(batch_i) + '.p')


