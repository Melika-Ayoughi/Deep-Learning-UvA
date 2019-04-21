"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch.nn as nn
import torch
import torch.optim as optim
import sys

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OPTIMIZER = 'SGD'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch


    Implement accuracy computation.
    """

    predicted_labels = np.argmax(predictions.detach().numpy(), axis=1)
    target_labels = targets.numpy()

    accuracy = (predicted_labels == target_labels).mean()


    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    dataset = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    a, b, c = dataset['train'].images.shape[1:]
    n_classes = dataset['train'].labels.shape[1]
    n_inputs = a * b * c


    mlp = MLP(n_inputs, dnn_hidden_units, n_classes)

    if (FLAGS.optimizer == 'SGD'):
        optimizer = optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = optim.Adam(mlp.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-2)

    crossentropy = nn.CrossEntropyLoss()

    test_input, test_labels = dataset['test'].images, dataset['test'].labels
    test_labels = np.argmax(test_labels, axis=1)
    test_input = np.reshape(test_input, (test_input.shape[0], n_inputs))
    test_input, test_labels = torch.from_numpy(test_input), torch.from_numpy(test_labels).long()

    max_accuracy = 0
    min_loss = 0
    for step in range(FLAGS.max_steps):
        input, labels = dataset['train'].next_batch(FLAGS.batch_size)
        labels = np.argmax(labels,axis=1)
        input = np.reshape(input, (FLAGS.batch_size, n_inputs))
        input, labels = torch.from_numpy(input), torch.from_numpy(labels).long()
        predictions = mlp.forward(input)

        loss = crossentropy(predictions, labels)
        # clean up old gradients
        mlp.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % FLAGS.eval_freq == 0):
            test_prediction = mlp.forward(test_input)
            test_loss = crossentropy(test_prediction, test_labels)
            test_accuracy = accuracy(test_prediction, test_labels)
            if (max_accuracy < test_accuracy):
                max_accuracy = test_accuracy
                min_loss = test_loss
            sys.stdout = open(str(FLAGS.dnn_hidden_units)+'_'+str(FLAGS.learning_rate)+'_'+str(FLAGS.max_steps)+'_'+str(FLAGS.batch_size)+'_'+str(FLAGS.batch_size)+'_'+str(FLAGS.optimizer)+'_mlp.csv', 'a')
            print("{},{:f},{:f}".format(step, test_loss, test_accuracy))

    sys.stdout = open(
        str(FLAGS.dnn_hidden_units) + '_' + str(FLAGS.learning_rate) + '_' + str(FLAGS.max_steps) + '_' + str(
            FLAGS.batch_size) + '_' + str(FLAGS.batch_size) + '_' + str(FLAGS.optimizer) + '_mlp.csv', 'a')
    print("max accuracy{:f}, minimum loss{:f}".format(max_accuracy, min_loss))

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help='The optimizer that is being used: either SGD or Adam')
    FLAGS, unparsed = parser.parse_known_args()

    main()