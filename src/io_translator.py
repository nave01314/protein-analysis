# This file will load and convert input and output of the program

import sys
import string
import res.helper
import numpy as np

# One-way dictionaries
letter_to_amino_map = {'A': 'Ala', 'B': 'Asx', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser', 'T': 'Thr', 'U': 'Selenocysteine', 'V': 'Val', 'W': 'Trp', 'X': 'Any', 'Y': 'Tyr', 'Z': 'Glx'}
letter_to_ss_map = {' ': 'Gap', 'H': 'Alpha helix', 'B': 'Beta bridge', 'E': 'Strand', 'G': 'Helix-3', 'I': 'Helix-5', 'T': 'Turn', 'S': 'Bend'}

# Primary dictionary is just map(ord/chr,list)-64 to get 1-26

# Secondary dicts are below
number_to_ss_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
ss_letter_to_number = inv_map = {v: k for k, v in number_to_ss_letter.items()}


def convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width):
    filename = res.helper.make_relative_path('res', 'fasta.txt')
    file = open(filename, 'r')
    sequences = []
    l_index = 0
    max_length = 0
    for line in file:
        if line.find('sequence') is not -1:
            sequences.append([])
            sequences[-1].append(line[:-1])   # Get rid of line breaks
            sequences[-1].append('')
            l_index = 1
        elif line.find('secstr') is not -1:
            sequences[-1].append(line[:-1])
            sequences[-1].append('')
            l_index = 3
        else:
            sequences[-1][l_index] = sequences[-1][l_index] + (line[:-1])

    for protein in sequences:
        if len(label) < width:
            label.append(protein[0][1:7])
            primary.append(protein[1])
            secondary.append(protein[3])
            if len(protein[1]) > max_length:
                max_length = len(protein[1])
        elif len(v_label) < v_width:
            v_label.append(protein[0][1:7])
            v_primary.append(protein[1])
            v_secondary.append(protein[3])
            if len(protein[1]) > max_length:
                max_length = len(protein[1])
    return max_length


def decode(bytes):
    return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '')


def prepare_primary_input(protein: str, pad_char: list, min_length: int):
    return list(map(lambda x: ord(x)-64, protein)) + pad_char * (min_length-len(protein))


def decode_primary_input(protein: str, pad_char: list):
    return ''.join(map(lambda x: chr(x+64).replace(chr(pad_char[0]+64), ' '), protein))


def prepare_secondary_input(protein: str, pad_char: list, min_length: int):
    return [0] + list(map(lambda x: int(ss_letter_to_number[x]), protein)) + pad_char * (min_length-len(protein))


def decode_secondary_input(protein: str):
    return ''.join(map(lambda x: number_to_ss_letter[str(x)], protein))


def letter_to_amino(letter: str):
    try:
        return letter_to_amino_map[letter]
    except KeyError:
        print('KeyError: %s is not an acceptable primary key!' % letter)


def letter_to_ss_type(letter: str):
    try:
        return letter_to_ss_map[letter]
    except KeyError:
        print('KeyError: %s is not an acceptable secondary key!' % letter)


def make_batch(input_data, protein_len, batch_size):
    batch_encoder = []
    for length_idx in range(protein_len):
        batch_encoder.append(np.array([input_data[batch_idx][length_idx] for batch_idx in range(batch_size)], dtype=np.int32))
    return batch_encoder


def undo_batch(batched_data):
    normal_data = []
    for batch_idx in range(len(batched_data[0])):
        normal_data.append([])
        for array in batched_data:
            normal_data[batch_idx].append(array[batch_idx])
    return normal_data
