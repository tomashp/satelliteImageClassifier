# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import logging

def load_label_names():
    return ['other', 'water', 'fores', 'city']

def load_batch(dataset_folder_path, batchName, batch_id):
    with open(dataset_folder_path + batchName + str(batch_id) +".pkl", mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data']
    labels = batch['labels']
        
    return features, labels

def display_stats(dataset_folder_path, batchName, batch_id, sample_id):
    features, labels = load_batch(dataset_folder_path, batchName, batch_id)
    
    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))
    
    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))
    
    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    
    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

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

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(dataset_folder_path, batchName, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches): # batch0 is a testing batch
        print("preprocess_and_save_data load batch ID:", batch_i)
        features, labels = load_batch(dataset_folder_path, batchName, batch_i)
        
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation], 
                             dataset_folder_path + 'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         dataset_folder_path + 'preprocess_validation.p')

    # load the test dataset
    with open(dataset_folder_path + batchName + '0' + ".pkl", mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data']
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         dataset_folder_path + 'preprocess_training.p')

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 5, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    conv3_bn = tf.layers.batch_normalization(conv3_pool)
    
    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)  

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)
    
    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)
    
    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)    
    
    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)        
    
    # 14
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=4, activation_fn=None)
    return out

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch, x, y, keep_prob):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })

def print_stats(session, feature_batch, label_batch, cost, accuracy, x, y, keep_prob, sess, valid_features, valid_labels):
    loss = sess.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = sess.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    logging.info('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size, dataset_folder_path):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = dataset_folder_path + 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

# interface for training CNN
#       dataset_folder_path -path to 
#       save_model_path - 
#       batchName - 
def run(dataset_folder_path, save_model_path, batchName):
    # # Explore the dataset - optional tu check if data sets are ok
    # batch_id = 3
    # sample_id = 3000
    # display_stats(dataset_folder_path, batchName, 0, sample_id)
    # display_stats(dataset_folder_path, batchName, 1, sample_id)
    # display_stats(dataset_folder_path, batchName, 2, sample_id)
    # display_stats(dataset_folder_path, batchName, 3, sample_id)
    logging.info('Running CNN training')
    preprocess_and_save_data(dataset_folder_path, batchName, normalize, one_hot_encode)
    valid_features, valid_labels = pickle.load(open(dataset_folder_path + 'preprocess_validation.p', mode='rb'))
    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 5), name='input_x')
    y =  tf.placeholder(tf.float32, shape=(None, 4), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    epochs = 10
    batch_size = 128
    keep_probability = 0.7
    learning_rate = 0.001

    logging.info('CNN training parameters: epoch: %s, batch size: %s, keep probability: %s, learning reate: %s', epochs, batch_size, keep_probability, learning_rate)

    logits = conv_net(x, keep_prob)
    model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    print('Training...')
    logging.info('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches):
                for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size, dataset_folder_path):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels, x, y, keep_prob)
                    
                print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i), end='')
                logging.info('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i))
                print_stats(sess, batch_features, batch_labels, cost, accuracy, x, y, keep_prob, sess, valid_features, valid_labels)
                
        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)