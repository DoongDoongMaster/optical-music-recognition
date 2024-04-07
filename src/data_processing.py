import numpy as np
from constant.common import EXP, FEATURE, MULTI_LABEL, PAD_STAVE, PNG
from constant.path import DATA_RAW_PATH, OSMD
from score2stave import Score2Stave
from util import Util


class DataProcessing:
    @staticmethod
    def process_all_score2stave():
        """
        raw/osmd 에 있는 모든 score 를 padding stave로 변환해서 img, csv로 저장
        """
        # osmd title paths 가져오기
        title_path_ = f"{DATA_RAW_PATH}/{OSMD}"
        title_path_list = Util.get_all_subfolders(title_path_)
        # title 마다 score2stave -- score가 한 장이 아니라 여러 장일 수 있으니까 반복문으로 처리
        for title_path in title_path_list:
            # 모든 score 불러와서 score2stave 후, padding 준 거 저장
            score_path_list = Util.get_all_files(f"{title_path}", EXP[PNG])
            for score_path in score_path_list:
                DataProcessing.process_score2stave(score_path)

    # feature 생성
    @staticmethod
    def process_score2stave(score_path):
        # -- score -> stave
        title = Util.get_title(score_path)

        stave_list = Score2Stave.transform_score2stave(score_path)

        # -- padding stave img
        pad_stave_list = Score2Stave.transform_stave2padStave(stave_list)

        # -- 1. image file로부터 feature 저장하려면 아래 코드
        # img_list = get_all_files(f"{DATA_FEATURE_PATH}/{title}", "png")
        # feature_list = transform_staveImg2feature(img_list)
        # -- 2. 위 feature 그대로 쓰려면 아래 코드
        feature_list = pad_stave_list

        # 데이터 이어붙이기
        merged_data = np.concatenate(feature_list, axis=1)

        # 전치
        transposed_data = np.transpose(merged_data)

        # -- save stave img, feature
        Score2Stave.save_stave_png(title, pad_stave_list, MULTI_LABEL, PAD_STAVE)
        Util.save_feature_csv(title, transposed_data, MULTI_LABEL, FEATURE)
