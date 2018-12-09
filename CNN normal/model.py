import pickle
import time
import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import re
import cifar10

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Explore the dataset
batch_id = 3
sample_id = 7000

# Hyper parameters
epochs = 3000
batch_size = 200
keep_probability = 1.0
learning_rate = 0.001

#file parameters
dataset_name = "default"
checkpoint_dir = "checkpoint"
use_distortion_for_training = True
data_dir = "../.."

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  
def load_preprocess_training_batch(batch_id, batch_size, subset, use_distortion_for_training, data_dir):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with tf.device('/cpu:0'):
        use_distortion = 'train' in subset and use_distortion_for_training
        dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size, epochs)
        return image_batch, label_batch

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

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
    
def conv2d(conv_filter, prev, name="conv2d"):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(prev, conv_filter, strides=[1,1,1,1], padding='SAME')
        conv = tf.nn.relu(conv)
        conv_pool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv_bn = tf.layers.batch_normalization(conv_pool)
        return conv_bn

def conv_net(x, keep_prob, reuse):
    with tf.variable_scope("model", reuse=reuse):
        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))
    
        # 1, 2
        conv1_bn = conv2d(conv1_filter, x, name="conv1")
    
        # 3, 4
        conv2_bn = conv2d(conv2_filter, conv1_bn, name="conv2")
    
        # 5, 6
        conv3_bn = conv2d(conv3_filter, conv2_bn, name="conv3")
    
        # 7, 8
        conv4_bn = conv2d(conv4_filter, conv3_bn, name="conv4")
    
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
        
        #14
        full5 = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=1024, activation_fn=tf.nn.relu)
        full5 = tf.nn.dropout(full5, keep_prob)
        full5 = tf.layers.batch_normalization(full5)
    
        # 15
        out = tf.contrib.layers.fully_connected(inputs=full5, num_outputs=10, activation_fn=None)
        return out

def model_dir():
    return "{}_{}".format(
        dataset_name, batch_size)
    
def save(sess, step):
    model_name = "cifar10Normal.model"
    check_dir = os.path.join(checkpoint_dir, model_dir())
  
    if not os.path.exists(check_dir):
      os.makedirs(check_dir)
  
    saver.save(sess,
            os.path.join(check_dir, model_name),
            global_step=step)

def load(sess):
    print(" [*] Reading checkpoints...")
    check_dir = os.path.join(checkpoint_dir, model_dir())
    
    ckpt = tf.train.get_checkpoint_state(check_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(check_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
  
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
    #show stats of a batch
    display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
    
    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    #Create tensors for training variables
    train_x, train_y = load_preprocess_training_batch(batch_id, batch_size, "train", use_distortion_for_training, data_dir)
    
    # Build model
    logits = conv_net(x, keep_prob, False)
    
    #Create tensors for validation variables
    valid_x, valid_y = load_preprocess_training_batch(batch_id, batch_size, "validation", use_distortion_for_training, data_dir)
    
    #Create saver
    saver = tf.train.Saver()
    
    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    print('Training...')
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # Initializing the variables
        sess.run(init_ops)
        
        best_accuracy = 0.0
        last_accuracy = 0.0
        best_epoch = 0
        total_time = 0.0
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = load(sess)
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")
          
        for epoch in range(epochs):
          
            batch_x, batch_y = sess.run([train_x, train_y])
            valid_batch_x, valid_batch_y = sess.run([valid_x, valid_y])
          
            sess.run(optimizer,
                         feed_dict={
                             x: batch_x,
                             y: batch_y,
                             keep_prob: keep_probability
                         })
  
            loss = sess.run(cost,
                         feed_dict={
                             x: batch_x,
                             y: batch_y,
                             keep_prob: 1.
                         })
  
            valid_acc = sess.run(accuracy,
                         feed_dict={
                             x: valid_batch_x,
                             y: valid_batch_y,
                             keep_prob: 1.
                         })
        
            total_time = time.time() - start_time
            last_accuracy = valid_acc
            print('Epoch: [%2d/%2d] time: %4.4f Loss: %4f Validation Accuracy: %4f' \
                  % (epoch + 1, epochs, total_time, loss, valid_acc))
            
            if best_accuracy < valid_acc:
                best_accuracy = valid_acc
                best_epoch = epoch + 1
            
            counter += 1
            if np.mod(counter, 200) == 0:
                save(sess, counter)
              
        #Show variables and their sizes
        show_all_variables()
      
        print("Best accuracy = %4f at epoch %2d" % (best_accuracy, best_epoch))
        print("Total time = %4.4f sec and accuracy = %4f perc" % (total_time, last_accuracy))