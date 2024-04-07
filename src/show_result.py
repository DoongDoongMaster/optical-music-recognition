import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from constant import (
    EXP,
    IMAGE_PATH,
    PITCH,
    PITCH_NOTE2CODE,
    PITCH_NOTES,
    PNG,
    PREDICT_STD,
)
from util import Util


class ShowResult:
    @staticmethod
    def show_label_dict_plot(label: dict[str, List[float]], start=0, end=None):
        """
        -- label 그래프
        {
            "HH": [1, 0, 0, ...],
            "ST": [0, 0, 0, ...],
            ...
        }
        """
        if end is None:  # end가 none이라면 y_true 끝까지
            end = len(label[list(label.keys())[0]])  # 첫 번째 value의 길이

        leng = len(PITCH_NOTE2CODE)
        for key, label_arr in label.items():
            data = np.array(label_arr)
            plt.subplot(leng, 1, PITCH_NOTE2CODE[key] + 1)
            plt.plot(data)
            plt.axis([start, end, 0, 1])
            plt.title(f"{key}")

        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = Util.get_datetime()
        plt.savefig(f"{IMAGE_PATH}/{PITCH}/pitch-predict-mat-{date_time}.{EXP[PNG]}")
        plt.show()

    @staticmethod
    def convert_to_sheet_music(prediction):
        sheet_music = []
        for note, probabilities in prediction.items():
            pitch_index = PITCH_NOTES.index(note)
            for i, prob in enumerate(probabilities):
                if prob == 1:
                    sheet_music.append((pitch_index, i))
        return sheet_music

    @staticmethod
    def plot_sheet_music(sheet_music, title):
        plt.figure(figsize=(10, 5))
        for pitch_index, time_index in sheet_music:
            plt.plot(
                [time_index, time_index + 1],
                [pitch_index, pitch_index],
                color="black",
                linewidth=2,
            )
        plt.yticks(range(len(PITCH_NOTES)), PITCH_NOTES)
        plt.xlabel("Time")
        plt.ylabel("Pitch")
        plt.title(title)
        plt.grid(True)
        # plt.xlim(0, max([time_index for _, time_index in sheet_music]) + 1)  # x 축 범위 설정
        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = Util.get_datetime()
        plt.savefig(f"{IMAGE_PATH}/{PITCH}/pitch-predict-sheet-{date_time}.png")
        plt.show()

    @staticmethod
    def get_predict2threshold(predict_data, n_classes) -> List[float]:
        # predict standard 이상일 때 1, else 0
        each_note_arr = predict_data

        for i in range(len(predict_data)):
            drums = []
            for j in range(n_classes):
                if predict_data[i][j] > PREDICT_STD:
                    drums.append(j)
                    each_note_arr[i][j] = 1
                else:
                    each_note_arr[i][j] = 0

        return each_note_arr