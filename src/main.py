# This file should be clear of functions, or loops. It should call other functions, time them, and give console output.

import io_translator
import p_predictor as predict
import res.helper
import test

label = []
primary = []
secondary = []
v_label = []
v_primary = []
v_secondary = []

print()
res.helper.print('Beginning FASTA load from file...')
max_length = io_translator.convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary)
res.helper.print('FASTA data loaded successfully...')

res.helper.print('Beginning primary sequence conversion...')
primary = io_translator.input_reformat(primary, max_length)
v_primary = io_translator.input_reformat(v_primary, max_length)
res.helper.print('Primary sequence conversion finished successfully...')

res.helper.print('Beginning secondary sequence conversion...')
secondary = io_translator.input_reformat(secondary, max_length)
v_secondary = io_translator.input_reformat(v_secondary, max_length)
res.helper.print('Secondary sequence conversion finished successfully...')

res.helper.print('Beginning model training...')
accuracy = predict.train(primary, secondary, v_primary, v_secondary)
res.helper.print('Model training finished with an accuracy of %s...' % accuracy)

# res.helper.print('Beginning model training...')
# loss = predict.train(primary, secondary)
# res.helper.print('Model training finished with a loss of %s...' % loss)
#
# res.helper.print('Beginning model validation...')
# success = predict.assess_model(v_primary, v_secondary)
# res.helper.print('Model validation finished with a success rate of %s...' % success)
#
# print('\nFinal Results: Model had a %s success rate in predicting secondary protein structure!' % (str(success)+'%'))




