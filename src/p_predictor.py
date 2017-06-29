# This file is for all ML functions

import numpy as np
import tensorflow as tf
import os

import io_translator as tr


def train(primary: list, secondary: list, v_primary: list, v_secondary: list, max_length: int):
    """Sequence-to-sequence model with an attention mechanism."""
    # see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
    # compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    input_vocab_size = 27                # Each amino is a character A-Z with a padding character ' '
    output_vocab_size = 8          # Each SS type is a character (see map in io_translator)
    buckets = [(200, 200)]           # Inputs and output sequences must be less than 10, respectively
    pad = [0]                      # Fill words shorter than a bucket size with padding
    batch_size = 1                # for parallel training (later) I don't know how to change this

    cell_size = 12                      # Fed to model (determines complexity of NN)
    model_layers = 1                 # Self-explanatory; how many "simple" cells are packaged together into the model
    learning_rate = 0.1            # Self-explanatory; used in optimizer

    # Convert data to numbers
    train_input = []
    train_targets = []
    train_weights = []
    for protein in primary:
        train_input.append(tr.prepare_primary_input(protein, pad, buckets[0][0]))
    train_input *= batch_size
    for protein in secondary:
        train_targets.append(tr.prepare_secondary_input(protein, pad, buckets[0][1]))
        # train_target_weights.append([1.0] * (len(protein)+1) + [0.0] * (buckets[0][1]-len(protein)-1))
        train_weights.append([1.0] * buckets[0][1])   # todo figure out
    train_targets *= batch_size
    train_weights *= batch_size

    # Move to batch-major formatting
    batched_train_input = tr.make_batch(train_input, buckets[0][0], batch_size * len(primary))
    batched_train_targets = tr.make_batch(train_targets, buckets[0][1], batch_size * len(secondary))
    batched_train_weights = tr.make_batch(train_weights, buckets[0][1], batch_size * len(secondary))

    # todo should likely have EOS \n and GO 1 symbols instead of 0

    class MySeq2Seq(object):
        def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.batch_size = batch_size

            cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
            if num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

            # The seq2seq function: we use embedding for the input and attention
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    feed_previous=do_decode)

            # Feeds for inputs
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in range(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one
            targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False))

            # Gradients update operation for training the model
            params = tf.trainable_variables()
            self.updates = []
            for b in range(len(buckets)):
                self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[b]))

            self.saver = tf.train.Saver(tf.global_variables())

        def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
            bucket_id = 0                   # todo: auto-select
            encoder_size, decoder_size = self.buckets[bucket_id]

            # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]

            # Since our targets are decoder inputs shifted by one, we need one more.
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size * len(primary) if not test else len(v_primary)], dtype=np.int32)

            # Output feed: depends on whether we're testing or not
            if not test:
                output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
            else:
                output_feed = [self.losses[bucket_id]]	    # Loss for this batch.
                for l in range(decoder_size):	            # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

            outputs = session.run(output_feed, input_feed)

            if not test:
                return outputs[0], outputs[1]               # Gradient norm, loss
            else:
                return outputs[0], outputs[1:]              # loss, outputs.

    def test():
        test_input = []
        test_targets = []
        test_weights = []
        for protein in v_primary:
            test_input.append(tr.prepare_primary_input(protein, pad, buckets[0][0]))
        test_input *= batch_size
        for protein in v_secondary:
            test_targets.append(tr.prepare_secondary_input(protein, pad, buckets[0][1]))
            test_weights.append([1.0] * buckets[0][1])    # todo figure out
        test_targets *= batch_size
        test_weights *= batch_size

        batched_input_test_data = tr.make_batch(test_input, buckets[0][0], batch_size * len(v_primary))
        batched_target_test_data = tr.make_batch(test_targets, buckets[0][1], batch_size * len(v_secondary))
        batched_weights_test_data = tr.make_batch(test_weights, buckets[0][1], batch_size * len(v_secondary))

        perplexity, outputs = my_model.step(session, batched_input_test_data, batched_target_test_data, batched_weights_test_data, test=True)
        words = tr.undo_batch(np.argmax(outputs, axis=2).tolist())   # shape (max, max, 256)

        correct_predictions = 0
        for i in range(len(test_input)):
            source_word = tr.decode_primary_input(test_input[i], pad)
            predicted_word = tr.decode_secondary_input(words[i])
            target_word = tr.decode_secondary_input(test_targets[i])[1:]

            if predicted_word == target_word:
                print('>> success >> Step %d Perplexity %f Primary: [%s] Predicted/Target SS: [%s]' % (step, perplexity, source_word, predicted_word))
                correct_predictions += 1
            else:
                print('Step %d Perplexity %f Primary: [%s] Predicted SS: [%s] Target SS: [%s]' % (step, perplexity, source_word, predicted_word, target_word))

        print('Model has %s%% accuracy' % (str(round(correct_predictions/len(test_input)*100, 2))))

        if correct_predictions == len(test_input):
            exit()

    step = 0
    test_step = 1
    with tf.Session() as session:
        my_model = MySeq2Seq(input_vocab_size, output_vocab_size, buckets,
                               size=cell_size, num_layers=model_layers, batch_size=batch_size)
        session.run(tf.global_variables_initializer())
        while True:
            my_model.step(session, batched_train_input, batched_train_targets, batched_train_weights, test=False)
            step = step + 1
            if step % test_step == 0:
                test()
