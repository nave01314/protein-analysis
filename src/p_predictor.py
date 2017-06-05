# This file is for all ML functions

import tensorflow as tf


def train(primary, secondary):
    return 0


def assess_model(primary, secondary):
    correct = 0
    # for seq in secondary:
    #     training_result = []
    #     for i in range(0, len(seq)-1):
    #         if training_result[i] is secondary[i]:
    #             correct += 1
    return correct/(len(secondary)-1)
