# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.
Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict
  
def read_pickle_from_file_BaW(filename, threshold, data_dir):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
	  
  X = data_dict[b"data"] 
  labels = data_dict[b"labels"]
  
  X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
  Y = np.zeros((10000, 32, 32, 1))
    
  #Visualizing CIFAR 10
  for i in range(len(X)):
      
      picture_array = X[i:i+1][0]
      print("picture_array: ", np.array(picture_array).shape)
      picture = Image.fromarray(np.array(picture_array), 'RGB')
      print("picture: ", np.array(picture).shape)
      gray_scale = picture.convert('L')
      print("gray_scale: ", np.array(gray_scale).shape)
      Y[i:i+1][0] = np.expand_dims(binarize_array(np.array(gray_scale), threshold=threshold), axis=-1)
      print("after: ", Y[i:i+1][0].shape)
      
      if i % 1000 == 0:
          f, axarr = plt.subplots(2)
          
          title = "%s , threshold= %d out of 255" % (get_label(labels[i]), threshold)
          f.suptitle(title, fontsize=12)
                
          axarr[0].set_axis_off()
          axarr[0].imshow(picture)
          axarr[0].set_title("RGB")
        
          axarr[1].set_axis_off()
          axarr[1].imshow(np.array(gray_scale), cmap="gray")
          axarr[1].set_title("Monochrome")
         
#          axarr[2].set_axis_off()
#          axarr[2].imshow(Y[i:i+1][0])
#          axarr[2].set_title("Black and White")
          
          # save the figure to file
          f.savefig("%s\\pictures%d\\%s.png" % (data_dir, threshold, get_label(labels[i])))
          plt.close(f)
			
  data_dict[b"data"] = Y
			
  return data_dict

def get_label(label):
    classes = {
            0: "airplane",
            1: "car",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
    }
    
    invalid = "Number %d is invalid" % (label)
    return classes.get(label, invalid)
        
  
def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())
		
def convert_to_tfrecord_BaW(input_files, output_file, threshold, data_dir):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file_BaW(input_file, threshold, data_dir)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
#  for mode, files in file_names.items():
#    input_files = [os.path.join(input_dir, f) for f in files]
#    output_file = os.path.join(data_dir, mode + '.tfrecords')
#    try:
#      os.remove(output_file)
#    except OSError:
#      pass
#    # Convert to tf.train.Example and write the to TFRecords.
#    convert_to_tfrecord(input_files, output_file)
	
    
  thresholds = [150]
  for threshold in thresholds: 
    for mode, files in file_names.items():
      mode = mode + "BaW" + str(threshold)
      input_files = [os.path.join(input_dir, f) for f in files]
      output_file = os.path.join(data_dir, mode + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      convert_to_tfrecord_BaW(input_files, output_file, threshold, data_dir)
  print('Done!')


if __name__ == '__main__':
#  parser = argparse.ArgumentParser()
#  parser.add_argument(
#      '--data-dir',
#      type=str,
#      default='',
#      help='Directory to download and extract CIFAR-10 to.')
#
#  args = parser.parse_args()
#  main(args.data_dir)
    main("../..")