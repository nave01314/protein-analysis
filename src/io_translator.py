# This file will load and convert input and output of the program

import sys
import string
import res.helper
import numpy

amino_map = {'1': 'Ala', '2': 'Asx', '3': 'Cys', '4': 'Asp', '5': 'Glu', '6': 'Phe', '7': 'Gly', '8': 'His', '9': 'Ile', '10': '10 is not a key!', '11': 'Lys', '12': 'Leu', '13': 'Met', '14': 'Asn', '15': '15 is not a key!', '16': 'Pro', '17': 'Gln', '18': 'Arg', '19': 'Ser', '20': 'Thr', '21': 'Selenocysteine', '22': 'Val', '23': 'Trp', '24': 'Any', '25': 'Tyr', '26': 'Glx'}


def convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width):
    filename = res.helper.make_relative_path('res', 'fasta.txt')
    file = open(filename, 'r')
    sequences = []
    l_index = 0
    max_length = 0
    for line in file:
        if line.find('sequence') is not -1:
            sequences.append([])
            sequences[len(sequences)-1].append(line[:-1])   # Get rid of line breaks
            sequences[len(sequences)-1].append('')
            l_index = 1
        elif line.find('secstr') is not -1:
            sequences[len(sequences)-1].append(line[:-1])
            sequences[len(sequences)-1].append('')
            l_index = 3
        else:
            sequences[len(sequences)-1][l_index] = sequences[len(sequences)-1][l_index] + (line[:-1])

    for sequence in sequences:
        if len(label) < width:
            label.append(sequence[0][1:7])
            primary.append(sequence[1])
            secondary.append(sequence[3])
        elif len(v_label) < v_width:
            v_label.append(sequence[0][1:7])
            v_primary.append(sequence[1])
            v_secondary.append(sequence[3])

        if len(sequence[1]) > max_length:
            max_length = len(sequence[1])

    return max_length


def input_reformat(sequences, width, length):
    new_format = numpy.full((width, length), -1, dtype=numpy.int8)
    countx = 0
    for seq in sequences:
        county = 0
        for char in seq:
            if char is not ' ':
                new_format[countx][county] = string.ascii_uppercase.index(char)
            county += 1
        countx += 1

    return new_format


def print_sequences(label, primary, secondary):
    for i in range(0, len(label)-1):
        print(label[i])
        print(primary[i])
        print(secondary[i])
        print()
