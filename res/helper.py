# Add global helper functions here

import os
import sys
from timeit import default_timer as timer
from difflib import SequenceMatcher


def make_relative_path(subdirectory, file):
    path = os.path.dirname(os.path.abspath(__file__))[:-4]
    return os.path.join(path, subdirectory, file)


def print(message):
    sys.stdout.write(str(round(timer(), 4)) + ': ' + message + os.linesep)


def assess_accuracy(predicted: str, target: str):
    return SequenceMatcher(None, predicted, target).ratio()
