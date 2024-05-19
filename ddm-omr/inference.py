
import os
from configs import getconfig
from produce_data.data_processing import DataProcessing

cofigpath = f"workspace/config.yaml"

args = getconfig(cofigpath)
DataProcessing.process_all_score2measure(args)