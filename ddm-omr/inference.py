import os
from configs import getconfig
from produce_data.data_processing import DataProcessing

import sys

sys.path.append("/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr")
from staff2score import StaffToScore

cofigpath = f"workspace/config.yaml"

args = getconfig(cofigpath)
# DataProcessing.process_all_score2measure(args)
staff2score = StaffToScore(args)

# staff2score.training()
staff2score.predict()
