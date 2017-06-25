# This file will load and convert input and output of the program

import sys
import string
import res.helper
import numpy

# One-way dictionaries
letter_to_amino_map = {'A': 'Ala', 'B': 'Asx', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser', 'T': 'Thr', 'U': 'Selenocysteine', 'V': 'Val', 'W': 'Trp', 'X': 'Any', 'Y': 'Tyr', 'Z': 'Glx'}
letter_to_ss_map = {' ': 'Gap', 'H': 'Alpha helix', 'B': 'Beta bridge', 'E': 'Strand', 'G': 'Helix-3', 'I': 'Helix-5', 'T': 'Turn', 'S': 'Bend'}

# Primary dictionary is just map(ord/chr,list)-65 to get 1-26

# Secondary dicts are below
number_to_ss_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
ss_letter_to_number = inv_map = {v: k for k, v in number_to_ss_letter.items()}


def convert_FASTA(label, primary, secondary, v_label, v_primary, v_secondary, width, v_width):
    filename = res.helper.make_relative_path('res', 'test.txt')
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


def letter_to_amino(letter: str):
    try:
        return letter_to_amino_map[letter]
    except KeyError:
        print('KeyError: %s is not an acceptable primary key!' % letter)


def letter_to_string(letter: str):
    try:
        return letter_to_ss_map[letter]
    except KeyError:
        print('KeyError: %s is not an acceptable secondary key!' % letter)


def convert_number_to_primary_letter(num: int):
    return chr(num+64)


def convert_primary_letter_to_number(letter: str):
    return ord(letter)-64


def convert_number_to_ss_letter(num: str):
    return number_to_ss_letter[str(num)]


def convert_ss_letter_to_number(letter: str):
    return ss_letter_to_number[letter]
