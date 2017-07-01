# This file should be clear of functions, or loops. It should call other functions, time them, and give console output.

import os
import tensorflow as tf

import io_translator as tr
import res.helper as helper
import p_predictor as predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_vocab_size = 27                # Each amino is a character A-Z with a padding character ' '
output_vocab_size = 8          # Each SS type is a character (see map in io_translator)
pad = [0]                      # Fill words shorter than a bucket size with padding

cell_size = 12                      # Fed to model (determines complexity of NN)
model_layers = 1                 # Self-explanatory; how many "simple" cells are packaged together into the model
learning_rate = 0.1            # Self-explanatory; used in optimizer

# buckets = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500), (600, 600), (700, 700), (800, 800), (900, 900), (1000, 1000), (1100, 1100), (1200, 1200)]
buckets = [(100, 100)]
batch_size = 1000

helper.print('Beginning FASTA load from file...')
data_set, v_data_set, labels, v_labels = tr.read_data_to_buckets('fasta.txt', buckets)
helper.print('FASTA data loaded training proteins and validation proteins successfully...')

helper.print('Creating model...')
model = predict.make_model(input_vocab_size, output_vocab_size, buckets, cell_size, model_layers, batch_size, learning_rate)

helper.print('Training model...')
predict.train(model, data_set, v_data_set, buckets, 0, batch_size, pad)
