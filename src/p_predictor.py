# This file is for all ML functions

import io_translator as translate


def train(primary: list, secondary: list, v_primary: list, v_secondary: list, max_length: int):
    """Sequence-to-sequence model with an attention mechanism."""
    # see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
    # compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py
    import numpy as np
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    vocab_size = 27                # Each amino is a character A-Z with a padding character ' '
    target_vocab_size = 8          # Each SS type is a character (see map in io_translator)
    buckets = [(10, 10)]           # Inputs and output sequences must be less than 10, respectively
    pad = [0]                      # Fill words shorter than a bucket size with padding
    batch_size = 10                # for parallel training (later) I don't know how to change this

    size = 12                      # Fed to model (determines complexity of NN)
    num_layers = 1                 # Self-explanatory; how many "simple" cells are packaged together into the model
    learning_rate = 0.1            # Self-explanatory; used in optimizer

    input_data = []
    for protein in primary:
        input_data = translate.prepare_primary_input(protein, pad, buckets[0][0]) * batch_size
    target_data = []
    target_weights = []
    for protein in secondary:
        target_data = translate.prepare_secondary_input(protein, pad, buckets[0][1]) * batch_size
        target_weights.append([1.0] * (len(protein)+1) + [0.0] * (buckets[0][1]-len(protein)-1))
    target_weights *= batch_size

    # EOS='\n' # end of sequence symbol todo use how?
    # GO=1		 # start symbol 0x01 todo use how?

    class BabySeq2Seq(object):
        def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.batch_size = batch_size

            cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
            if num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    feed_previous=do_decode)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in range(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one. OK
            targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False))

            # Gradients update operation for training the model.
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

            # Output feed: depends on whether we do a backward step or not.
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
        perplexity, outputs = my_model.step(session, input_data, target_data, target_weights, test=True)
        words = np.argmax(outputs, axis=2)  # shape (max, max, 256)
        source_word = translate.decode_primary_input(input_data[0], pad)
        predicted_word = translate.decode_secondary_input(words[0])
        target_word = translate.decode_secondary_input(target_data[0])
        print('step %d, perplexity %f, Primary: [%s] Secondary: [%s]' % (step, perplexity, source_word, predicted_word))
        if predicted_word == target_word:
            print('>>>>> success! Primary: [%s] Secondary: [%s] <<<<<<<' % (source_word, predicted_word))
            exit()

    step = 0
    test_step = 1
    with tf.Session() as session:
        my_model = BabySeq2Seq(vocab_size, target_vocab_size, buckets,
                               size=size, num_layers=num_layers, batch_size=batch_size)
        session.run(tf.global_variables_initializer())
        while True:
            test()
            my_model.step(session, input_data, target_data, target_weights, test=False)    # no outputs in training
            step = step + 1
            if step % test_step == 0:
                test()


def model():
    """Sequence-to-sequence model with an attention mechanism."""
    # see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
    # compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py
    import numpy as np
    import tensorflow as tf

    vocab_size=256 # We are lazy, so we avoid fency mapping and just use one *class* per character/byte
    target_vocab_size=vocab_size
    learning_rate=0.1
    buckets=[(10, 10)] # our input and response words can be up to 10 characters long
    PAD = [0] # fill words shorter than 10 characters with 'padding' zeroes
    batch_size=10 # for parallel training (later)

    input_data    = [list(map(ord, "hello")) + PAD * 5] * batch_size
    target_data   = [list(map(ord, "world")) + PAD * 5] * batch_size
    target_weights= [[1.0]*6 + [0.0]*4] *batch_size # mask padding. todo: redundant --

    # EOS='\n' # end of sequence symbol todo use how?
    # GO=1		 # start symbol 0x01 todo use how?


    class BabySeq2Seq(object):

        def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
            self.buckets = buckets
            self.batch_size = batch_size
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size

            cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
            if num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    feed_previous=do_decode)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in range(buckets[-1][0]):	# Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one. OK
            targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False))

            # Gradients update operation for training the model.
            params = tf.trainable_variables()
            self.updates=[]
            for b in range(len(buckets)):
                self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[b]))

            self.saver = tf.train.Saver(tf.all_variables())

        def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
            bucket_id=0 # todo: auto-select
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

            # Output feed: depends on whether we do a backward step or not.
            if not test:
                output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
            else:
                output_feed = [self.losses[bucket_id]]	# Loss for this batch.
                for l in range(decoder_size):	# Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

            outputs = session.run(output_feed, input_feed)
            if not test:
                return outputs[0], outputs[1]# Gradient norm, loss
            else:
                return outputs[0], outputs[1:]# loss, outputs.

    def decode(bytes):
        return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '')

    def test():
        perplexity, outputs = model.step(session, input_data, target_data, target_weights, test=True)
        words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
        word = decode(words[0])
        print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))
        if word == "world":
            print(">>>>> success! hello " + word + "! <<<<<<<")
            exit()

    step=0
    test_step=1
    with tf.Session() as session:
        model= BabySeq2Seq(vocab_size, target_vocab_size, buckets, size=10, num_layers=1, batch_size=batch_size)
        session.run(tf.initialize_all_variables())
        while True:
            model.step(session, input_data, target_data, target_weights, test=False) # no outputs in training
            if step % test_step == 0:
                test()
            step=step+1