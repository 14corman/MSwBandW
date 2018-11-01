# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:03:13 2018

@author: Cory Kromer-Edwards

"""

from PIL import Image
from scipy.misc import imsave
import numpy as np
import pickle
import matplotlib.pyplot as plt


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    return image


def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",
                        dest="input",
                        help="read this file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="write binarized file hre",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--threshold",
                        dest="threshold",
                        default=200,
                        type=int,
                        help="Threshold when to show white")
    return parser


if __name__ == "__main__":
    #args = get_parser().parse_args()
    with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    X = dict[b"data"] 
    Y = dict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    
    #Visualizing CIFAR 10
    fig, axes1 = plt.subplots(6,6,figsize=(5,5))
    for j in range(6):
        for k in range(0, 6, 2):
            i = np.random.choice(range(len(X)))
            picture_array = X[i:i+1][0]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(picture_array)
            
            picture = Image.fromarray(np.array(picture_array), 'RGB')
            picture = picture.convert('L')
            
            axes1[j][k+1].set_axis_off()
            axes1[j][k+1].imshow(binarize_array(np.array(picture), threshold=75))
        
    #Keys = labels, data, filenames
    
    
    #binarize_image(args.input, args.output, args.threshold)
    
    
    
    