# This file should be clear of functions, or loops. It should call other functions, time them, and give console output.

import io_translator
import p_predictor as predict
import res.helper
import numpy

label = []
primary = []
secondary = []
v_label = []
v_primary = []
v_secondary = []

width = 10
v_width = 0

encoder_inputs = []

encoder_inputs.append(list([1,2,3,4,5]))
encoder_inputs.append(list([6,7,8,9,10]))
encoder_inputs.append(list([11,12,13,14,15]))

print(encoder_inputs)

encoder_size = 5
batch_size = 3
batch_encoder_inputs = []

for length_idx in range(encoder_size):
    batch_encoder_inputs.append(numpy.array([encoder_inputs[batch_idx][length_idx] for batch_idx in range(batch_size)]))

for i in range(len(batch_encoder_inputs)):
    print(batch_encoder_inputs[i])

encoder_inputs = []
print(len(batch_encoder_inputs[0]))
print(len(batch_encoder_inputs))
for index in range(len(batch_encoder_inputs[0])):
    encoder_inputs.append([])
    for array in batch_encoder_inputs:
        encoder_inputs[index].append(array[index])

print(encoder_inputs)


res.helper.print('Beginning FASTA load from file...')
max_length = io_translator.convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width)
res.helper.print('FASTA data loaded %s training proteins and %s validation proteins successfully...' % (len(label),
                                                                                                        len(v_label)))
res.helper.print('Beginning model training...')
#predict.model()
predict.train(primary, secondary, v_primary, v_secondary, max_length)
#res.helper.print('Model training finished with an accuracy of %s...' % accuracy)

# res.helper.print('Beginning model training...')
# loss = predict.train(primary, secondary)
# res.helper.print('Model training finished with a loss of %s...' % loss)
#
# res.helper.print('Beginning model validation...')
# success = predict.assess_model(v_primary, v_secondary)
# res.helper.print('Model validation finished with a success rate of %s...' % success)
#
# print('\nFinal Results: Model had a %s success rate in predicting secondary protein structure!' % (str(success)+'%'))




