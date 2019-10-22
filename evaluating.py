import tensorflow as tf
import pickle
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import confusionMatrixDisp as confMtx

batch_size = 64
n_samples = 10
top_n_predictions = 5
log_path = 'allDataValidLogs'
allDataPredictions = {}
labelsPredictions = []
labelsList = []

def batch_features_labels(features, labels, imgNames, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end], imgNames[start:end]


def process_and_save_predictions( arrayWithPredictions, testLabel, imgNames ):
    for id in range(0 , len(arrayWithPredictions)):
        labelIdx = np.where (arrayWithPredictions[id] == np.amax(arrayWithPredictions[id]) )  #gets indx ( class ) of prediction
        allDataPredictions.update({imgNames[id] : labelIdx[0][0]})
        labelsPredictions.append(labelIdx[0][0])
        labelIdx = np.where (testLabel[id] == np.amax(testLabel[id]) )  #gets indx ( class ) of prediction
        labelsList.append(labelIdx[0][0])


def test_model( save_model_path, allDataPreProcessedPath ):
    
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        file_writer = tf.summary.FileWriter(log_path, loaded_graph )

        logging.info('ALL DATA: Evaluating...')
        #loop that iterates over preprocessedAllDataBatch
        for filename in os.listdir(allDataPreProcessedPath):
            if filename.endswith(".p"):
                with open(allDataPreProcessedPath + filename, mode='rb') as file:
                    batch = pickle.load(file, encoding='latin1')

                test_features = batch[0]
                logging.info('ALL DATA: Test features size is: %s', len(batch[0]) )
                test_labels = batch[1]
                logging.info('ALL DATA: Test labels size is: %s', len(batch[1]) )
                imgNames = batch[2]
                logging.info('ALL DATA: Img names size is: %s', len(batch[2]) )

                for train_feature_batch, train_label_batch, imgNames in batch_features_labels(test_features, test_labels, imgNames, batch_size):
                    # saving each prediction to dictionery to be able to repaint map
                    predArray = loaded_logits.eval(feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
                    process_and_save_predictions(predArray, train_label_batch, imgNames)

                    test_batch_acc_total += sess.run(
                        loaded_acc,
                        feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
                    test_batch_count += 1
                    print( filename, " batch step: ", test_batch_count)

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
        logging.info('ALL DATA: Testing Accuracy: %s', (test_batch_acc_total/test_batch_count) )

        confusion = tf.math.confusion_matrix(labelsList, labelsPredictions, num_classes= 4)
        print('Confusion Matrix:\n', confusion.eval(session=sess))
        logging.info('ALL DATA: Confusion Matrix:\n %s', confusion.eval(session=sess) )

    f = open("allDataPredictionsTEST.pkl","wb")
    pickle.dump(allDataPredictions,f)
    f.close()

