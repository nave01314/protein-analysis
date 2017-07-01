# This file will load and convert input and output of the program

import sys
import res.helper
import numpy as np
import random

# One-way dictionaries
letter_to_amino_map = {'A': 'Ala', 'B': 'Asx', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser', 'T': 'Thr', 'U': 'Selenocysteine', 'V': 'Val', 'W': 'Trp', 'X': 'Any', 'Y': 'Tyr', 'Z': 'Glx'}
letter_to_ss_map = {' ': 'Gap', 'H': 'Alpha helix', 'B': 'Beta bridge', 'E': 'Strand', 'G': 'Helix-3', 'I': 'Helix-5', 'T': 'Turn', 'S': 'Bend'}

# Primary dictionary is just map(ord/chr,list)-64 to get 1-26

# Secondary dicts are below
number_to_ss_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
ss_letter_to_number = inv_map = {v: k for k, v in number_to_ss_letter.items()}


def convert_raw_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width):
    filename = res.helper.make_relative_path('res', 'fasta.txt')
    file = open(filename, 'r')
    sequences = []
    l_index = 0
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
        elif len(v_label) < v_width:
            v_label.append(protein[0][1:7])
            v_primary.append(protein[1])
            v_secondary.append(protein[3])


def read_data_to_buckets(data_name: str, buckets: list):
    filename = res.helper.make_relative_path('res', data_name)
    file = open(filename, 'r')
    sequences = []
    l_index = 0
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

    labels = [[] for bucket in buckets]
    data_set = [[] for bucket in buckets]
    v_labels = [[] for bucket in buckets]
    v_data_set = [[] for bucket in buckets]
    count = 0
    for protein in sequences:
        if count < 300000:
            for bucket_id in range(len(buckets)):
                if len(protein[1]) < buckets[bucket_id][0]:
                    labels[bucket_id].append(protein[0][1:7])
                    data_set[bucket_id].append([protein[1], protein[3]])
                    break
        elif count < 350000:
            for bucket_id in range(len(buckets)):
                if len(protein[1]) < buckets[bucket_id][0]:
                    v_labels[bucket_id].append(protein[0][1:7])
                    v_data_set[bucket_id].append([protein[1], protein[3]])
                    break
        count += 1

    return data_set, v_data_set, labels, v_labels


def get_converted_batch(data_set: list, buckets: int, bucket_id: int, batch_size: int, pad: list):
    p_inputs = []
    s_inputs = []
    for i in range(batch_size):
        source, target = random.choice(data_set[bucket_id])
        p_inputs.append(source)
        s_inputs.append(target)

    # todo should likely have EOS \n and GO 1 symbols instead of 0
    train_inputs = []
    train_targets = []
    for protein in p_inputs:
        train_inputs.append(prepare_primary_input(protein, pad, buckets[0][bucket_id]))
    for protein in s_inputs:
        train_targets.append(prepare_secondary_input(protein, pad, buckets[0][1]))

    batched_p_inputs = []
    batched_s_inputs = []
    batched_weights = []
    for length_idx in range(buckets[bucket_id][0]):
        batched_p_inputs.append([train_inputs[batch_idx][length_idx] for batch_idx in range(batch_size)])
    for length_idx in range(buckets[bucket_id][1]):
        batched_s_inputs.append([train_targets[batch_idx][length_idx] for batch_idx in range(batch_size)])
        batched_weights.append([1.0] * batch_size)

    return batched_p_inputs, batched_s_inputs, batched_weights


def prepare_primary_input(protein: str, pad_char: list, min_length: int):
    return list(map(lambda x: ord(x)-64, protein)) + pad_char * (min_length-len(protein))


def decode_primary_input(protein: str, pad_char: list):
    return ''.join(map(lambda x: chr(x+64).replace(chr(pad_char[0]+64), ' '), protein))


def prepare_secondary_input(protein: str, pad_char: list, min_length: int):
    return [0] + list(map(lambda x: int(ss_letter_to_number[x]), protein)) + pad_char * (min_length-len(protein))


def decode_secondary_input(protein: str):
    return ''.join(map(lambda x: number_to_ss_letter[str(x)], protein))


def make_batch(input_data: list, protein_len, batch_size):
    batch_encoder = []
    for length_idx in range(protein_len):
        batch_encoder.append(np.array([input_data[batchidx][length_idx] for batchidx in range(batch_size)]))
    return batch_encoder


def undo_batch(batched_data: list):
    normal_data = []
    for batch_idx in range(len(batched_data[0])):
        normal_data.append([])
        for array in batched_data:
            normal_data[batch_idx].append(array[batch_idx])
    return normal_data





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



