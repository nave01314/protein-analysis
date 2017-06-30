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

width = 300000
v_width = 50000

res.helper.print('Beginning FASTA load from file...')
io_translator.convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width)
res.helper.print('FASTA data loaded %s training proteins and %s validation proteins successfully...' % (len(label),
                                                                                                        len(v_label)))

res.helper.print('Beginning model training...')
predict.train(primary, secondary, v_primary, v_secondary)


