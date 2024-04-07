import os
import cv2
import numpy as np


from constant import (
    DATA_FEATURE_PATH,
    EXP,
    MULTI_LABEL,
    PAD_STAVE,
    PNG,
    STAVE_HEIGHT,
    STAVE_WIDTH,
)
from util import Util


class Score2Stave:
    @staticmethod
    def transform_img2binaryImg(img):
        """
        rgb img -> binary img
        input : rgb image
        return : binary image
        """
        img = cv2.imread(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # -- 설정 임곗값(retval), 결과 이미지(biImg)
        ret, biImg = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY_INV
        )  # -- 임곗값을 초과할 경우 0, 아닐 경우 maxval
        return biImg

    @staticmethod
    def get_score_data(score_path):
        """
        score의 binary img, size 얻기
        return: binary img, height, width
        """
        biImg = Score2Stave.transform_img2binaryImg(score_path)
        height, width = biImg.shape
        return biImg, height, width

    @staticmethod
    def extract_segment_from_score(biImg):
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

    @staticmethod
    def extract_stave_from_score(biImg, score_width, cnt, stats):
        """
        score에서 stave 추출
        """
        stave_list = []
        # -- idx 0은 배경이라 제외
        for i in range(1, cnt):
            x, y, w, h, _ = stats[i]
            # -- stave 인식
            if (
                w > score_width * 0.1
            ):  # -- 주로 마지막 stave인 경우, 한 마디인 경우가 있을 수 있음.
                stave = biImg[y : y + h, x : x + w]
                stave_list.append(stave)

        return stave_list

    @staticmethod
    def transform_score2stave(score_path):
        """
        score로부터 stave image추출
        """

        biImg, _, width = Score2Stave.get_score_data(score_path)

        cnt, _, stats, _ = Score2Stave.extract_segment_from_score(biImg)

        stave_list = Score2Stave.extract_stave_from_score(biImg, width, cnt, stats)

        return stave_list

    @staticmethod
    def transform_stave2padStave(stave_list):
        pad_stave = []
        for stave in stave_list:
            pad_image = np.zeros((STAVE_HEIGHT, STAVE_WIDTH))

            # 새로운 이미지에 주어진 이미지 삽입
            pad_image[: stave.shape[0], : stave.shape[1]] = stave

            # 부족한 부분 0으로 채우기
            pad_image[stave.shape[0] :, :] = 0  # 높이 부족 부분
            pad_image[:, stave.shape[1] :] = 0  # 너비 부족 부분
            pad_stave.append(pad_image)
        return pad_stave

    @staticmethod
    def transform_staveImg2feature(img_list):
        """
        (선택)
        pad image file로부터 feature 추출
        단, padding 된 png만 (pad-stave 이름 붙은 것들만)
        """
        print("--- pad image file로부터 feature 추출 ---")

        feature_list = []
        for idx, img in enumerate(img_list):
            if PAD_STAVE in img:
                biImg = Score2Stave.transform_img2binaryImg(img)
                feature_list.append(biImg)
                print(f"{idx} --shape: {biImg.shape}")
        return feature_list

    @staticmethod
    def save_stave_png(title, stave_list, label_type, state):
        """
        save stave list
        """
        os.makedirs(f"{DATA_FEATURE_PATH}/{label_type}/{title}/", exist_ok=True)
        for idx, stave in enumerate(stave_list):
            date_time = Util.get_datetime()
            cv2.imwrite(
                f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}_{state}_{idx}_{date_time}.{EXP[PNG]}",
                stave,
            )
            print(state, idx, "--shape: ", stave.shape)
