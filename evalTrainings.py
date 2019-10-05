import tensorflow as tf
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
import confusionMatrixDisp as confMtx

batch_size = 64
n_samples = 10
top_n_predictions = 5
log_path = 'validationLogs'
labelsPredictions = []
labelsList = []

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def process_and_save_predictions( arrayWithPredictions , testLabel):
    for id in range(0 , len(arrayWithPredictions)):
        labelIdx = np.where (arrayWithPredictions[id] == np.amax(arrayWithPredictions[id]) )  #gets indx ( class ) of prediction
        labelsPredictions.append(labelIdx[0][0])
        labelIdx = np.where (testLabel[id] == np.amax(testLabel[id]) )  #gets indx ( class ) of prediction
        labelsList.append(labelIdx[0][0])
            
def test_model( save_model_path, filePath ):
    test_features, test_labels = pickle.load(open( filePath + 'preprocess_training.p', mode='rb'))
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

        for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, batch_size):
            # saving each prediction to dictionery to be able to repaint map
            predArray = loaded_logits.eval(feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            process_and_save_predictions(predArray, train_label_batch)

            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

            #file_writer.add_summary( (test_batch_acc_total/test_batch_count) , test_batch_count)

        print('TRAINING data -> Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
        logging.info('TRAINING data -> Testing Accuracy: %s', (test_batch_acc_total/test_batch_count) )

        confusion = tf.math.confusion_matrix(labelsList, labelsPredictions, num_classes= 4)
        print('Confusion Matrix:\n', confusion.eval(session=sess))
        logging.info('Confusion Matrix:\n %s', confusion.eval(session=sess) )
        #confMtx._test_cm( confusion, "Zbiór testowy" )
