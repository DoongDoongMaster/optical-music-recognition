import os
import cv2
from util import Util

from process_data.image2augment import Image2Augment
from produce_data.score2measure import Score2Measure


class SheetToScore(object):
    def __init__(self, args):
        self.args = args

    def extract_segment_from_score(self, biImg):
        """
        score에서 각 segment 추출
        객체 정보를 함께 반환하는 레이블링 함수
        cnt : 객체 수 + 1 (배경 포함)
        labels : 객체에 번호가 지정된 레이블 맵
        stats : N x 5, N은 객체 수 + 1이며 각각의 행은 번호가 지정된 객체를 의미, 5열에는 x, y, width, height, area 순으로 정보가 담겨 있습니다. x,y 는 좌측 상단 좌표를 의미하며 area는 면적, 픽셀의 수를 의미합니다.
        centroids : N x 2, 2열에는 x,y 무게 중심 좌표가 입력되어 있습니다. 무게 중심 좌표는 픽셀의 x 좌표를 다 더해서 갯수로 나눈 값입니다. y좌표도 동일합니다.
        """
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(biImg)
        return cnt, labels, stats, centroids

    def extract_stave_from_score(self, biImg, cnt, stats):
        """
        stave 추출
        악보
        """
        score_h, score_w = biImg.shape
        PAD = 10

        stave_list = []
        # -- idx 0은 배경이라 제외
        # 1. 임의 widht 이상일 때 stave라고 판단
        for i in range(1, cnt):
            x_s, y_s, w_s, h_s, _ = stats[i]

            x = round(x_s - PAD)

            t_y = round((y_s - PAD))
            y = max(t_y, 0)

            h = h_s + 2 * PAD

            w = round(w_s + 2 * PAD)

            # -- stave 인식
            # -- stave width가 score width와 같지 않은 경우가 있을 수도 있음
            if w >= score_w * 0.3:
                stave = biImg[y : y + h, x : x + w]
                stave_list.append(stave)

        result_stave_list = []
        min_h = score_h / len(stave_list) / 2
        # 1. stave라고 판단된 것 중에 임의 height 이상일 때 stave라고 판단
        for stave in stave_list:
            h, _ = stave.shape
            if h >= min_h:
                result_stave_list.append(stave)
        return result_stave_list

    def save_stave(self, title, stave_list):
        """
        save stave list
        """
        os.makedirs(
            f"{self.args.filepaths.feature_path.base}/stave/{title}", exist_ok=True
        )
        for idx, stave in enumerate(stave_list):
            date_time = Util.get_datetime()
            cv2.imwrite(
                f"{self.args.filepaths.feature_path.base}/stave/{title}/{title}-stave_{idx+1}_{date_time}.png",
                stave,
            )
            print(idx + 1, "--shape: ", stave.shape)

    def transform_score2stave(self, score_path):
        """
        score로부터 stave image추출
        """

        biImg = Image2Augment.readimg(score_path)

        # 배경이 검정색인 binary image일 때 잘 추출하더라
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list

    def sheet2stave(self):
        score_path = f"{self.args.filepaths.raw_path.osmd}/2002-1/2002-1.png"  # -- stave를 추출할 악보

        stave_list = self.transform_score2stave(score_path)
        self.save_stave("2002-1", stave_list)
