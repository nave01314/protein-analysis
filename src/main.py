# This file should be clear of functions, or loops. It should call other functions, time them, and give console output.

import io_translator
import p_predictor as predict
import res.helper

label = []
primary = []
secondary = []
v_label = []
v_primary = []
v_secondary = []

width = 1
v_width = 0

print()
res.helper.print('Beginning FASTA load from file...')
max_length = io_translator.convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width)
res.helper.print(str(primary))
res.helper.print(str(secondary))
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




