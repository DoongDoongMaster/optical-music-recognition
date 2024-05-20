import os
from process_data.image2augment import Image2Augment
from configs import getconfig
from produce_data.data_processing import DataProcessing

import sys

sys.path.append("/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr")
from staff2score import StaffToScore

cofigpath = f"workspace/config.yaml"

args = getconfig(cofigpath)
# DataProcessing.process_all_score2measure(args)
staff2score = StaffToScore(args)

staff2score.training()
staff2score.test()

# x_raw_path = [
#     f"../data/test/Rock-ver_measure_02_2024-05-19_05-31-40.png",
#     f"../data/test/test_img.png",
# ]
# biImg_list = []

# for x_raw in x_raw_path:
#     biImg = Image2Augment.readimg(x_raw)
#     biImg = 255 - biImg
#     biImg_list.append(biImg)

# staff2score.predict(biImg_list)
