import os 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
from collections import Counter 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers

import glob
import re

import albumentations as alb
from albumentations.pytorch import ToTensorV2

# from model import TrOMR

char_to_int_mapping = [
    "|",                # 0
    "barline",          # 1
    "clef-percussion",  # 2
    "note-eighth",      # 3
    "note-eighth.",     # 4
    "note-half",        # 5
    "note-half.",       # 6
    "note-quarter",     # 7
    "note-quarter.",    # 8
    "note-16th",   # 9
    "note-16th.",  # 10
    "note-whole",       # 11
    "note-whole.",      # 12
    "rest_eighth",      # 13
    "rest_eighth.",     # 14
    "rest_half",        # 15
    "rest_half.",       # 16
    "rest_quarter",     # 17
    "rest_quarter.",    # 18
    "rest_16th",   # 19
    "rest_16th.",  # 20
    "rest_whole",       # 21
    "rest_whole.",      # 22
    "timeSignature-4/4" # 23
]

pitch_to_int_mapping = [
    "|",  #1
    "nonote",#2
    "note-D4",#3
    "note-E4",#4
    "note-F4",#5
    "note-G4",#6
    "note-A4",#7
    "note-B4",#8
    "note-C5",#9
    "note-D5",#10
    "note-E5",#11
    "note-F5",#12
    "note-G5",#13
    "note-A5",#14
    "note-B5",#15
]


# 문자를 숫자로 변환
char_to_num = layers.StringLookup(
    vocabulary=list(char_to_int_mapping), mask_token=None
)
# print(char_to_num.get_vocabulary())
# test=["clef-percussion","note-eighth","|","note-eighth", "note-eighth", "note-eighth|note-eighth", "rest_whole"]
# print(char_to_num(test))
pitch_char_to_num = layers.StringLookup(
    vocabulary=list(pitch_to_int_mapping), mask_token=None
)

# 숫자를 문자로 변환
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
num_to_pitch_char = layers.StringLookup(
    vocabulary=pitch_char_to_num.get_vocabulary(), mask_token=None, invert=True
)

class StaffToScore(object):
    def __init__(self, args):
        self.args = args
        self.size_h = args.max_height
        # self.model = TrOMR(args)




if __name__ == "__main__":
    from configs import getconfig


    # args = getconfig("./workspace/config.yaml")
    # handler = StaffToScore(args)
    # predrhythm, predpitch, predlift = handler.predict(
    #     "../examples/test2/dark_1962926-44.jpg"
    # )
    # print(predrhythm)
    # print(predpitch)
    # print(predlift)
