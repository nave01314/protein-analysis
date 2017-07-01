import os
import numpy as np
import tensorflow as tf

import io_translator as tr
import res.helper as helper


def make_model(input_vocab_size, output_vocab_size, buckets, cell_size, model_layers, batch_size, learning_rate):
    class MySeq2Seq(object):
        def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size, learning_rate):
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
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

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
    # return MySeq2Seq(input_vocab_size, output_vocab_size, buckets,
    #                  size=cell_size, num_layers=model_layers, batch_size=batch_size)
    return MySeq2Seq(input_vocab_size, output_vocab_size, buckets=buckets,
                     size=cell_size, num_layers=model_layers, batch_size=batch_size, learning_rate=learning_rate)


def train(my_model, data_set, v_data_set, buckets, bucket_id, batch_size, pad):

    def test():
        perplexity, outputs = my_model.step(session, batched_v_primary, batched_v_secondary, batched_v_weights, test=True)
        normalized_sources = tr.undo_batch(batched_v_primary)
        words = tr.undo_batch(np.argmax(outputs, axis=2))  # shape (max, max, 256)
        normalized_targets = tr.undo_batch(batched_v_secondary)

        correctness = 0
        for i in range(len(normalized_sources)):
            source_word = tr.decode_primary_input(normalized_sources[i], pad)
            predicted_word = tr.decode_secondary_input(words[i])
            target_word = tr.decode_secondary_input(normalized_targets[i])[1:]

            accuracy = helper.assess_accuracy(predicted_word, target_word)
            correctness += accuracy

            if accuracy == 1.0:
                print('>> success >> Step %d Perplexity %f, Primary [%s] Predicted/Target [%s]' % (step, perplexity, source_word, predicted_word))
            else:
                print('Acc %f Step %d Perplexity %f Primary [%s] Predicted [%s] Target [%s]' % (accuracy, step, perplexity, source_word, predicted_word, target_word))

        print('Model is %s%% effective' % str(round(correctness/len(normalized_sources)*100)))

        if correctness == len(normalized_sources):
            exit()

    step = 0
    test_step = 5
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        while True:
            batched_primary, batched_secondary, batched_weights = tr.get_converted_batch(data_set, buckets, bucket_id, batch_size, pad)
            my_model.step(session, batched_primary, batched_secondary, batched_weights, test=False)    # no outputs in training
            step = step + 1
            if step % test_step == 0:
                batched_v_primary, batched_v_secondary, batched_v_weights = tr.get_converted_batch(v_data_set, buckets, 0, batch_size, pad)
                test()
