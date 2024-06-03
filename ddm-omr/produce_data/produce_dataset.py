import json
import math
import os
import re
import cv2
import sys

sys.path.append("/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr")
from xml2annotation import Xml2Annotation
from process_data.image2augment import Image2Augment

from util import Util


class ProduceDataset(object):
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
        cursor_list = []
        # -- idx 0은 배경이라 제외
        # 1. 임의 widht 이상일 때 stave라고 판단
        for i in range(1, cnt):
            x_s, y_s, w_s, h_s, _ = stats[i]

            x = math.floor(x_s - PAD)

            t_y = math.floor(y_s - PAD)
            y = max(t_y, 0)

            h = h_s + 2 * PAD

            w = math.floor(w_s + 2 * PAD)

            # -- stave 인식
            # -- stave width가 score width와 같지 않은 경우가 있을 수도 있음
            if w >= score_w * 0.2:
                stave = biImg[y : y + h, x : x + w]
                stave_list.append(stave)
                cursor = [x, y, w, h]
                cursor_list.append(cursor)  # stave 위치 저장

        result_stave_list = []
        result_cursor_list = []
        min_h = score_h / len(stave_list) / 2
        # 1. stave라고 판단된 것 중에 임의 height 이상일 때 stave라고 판단
        for idx, stave in enumerate(stave_list):
            h, _ = stave.shape
            if h >= min_h:
                result_stave_list.append(stave)
                result_cursor_list.append(cursor_list[idx])
        return result_stave_list, result_cursor_list

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

        (h, w) = biImg.shape[:2]
        target_width = self.args.score_width

        # 비율 계산
        ratio = target_width / float(w)
        target_height = int(h * ratio)

        # 이미지 리사이즈
        biImg = cv2.resize(
            biImg, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        # 배경이 검정색인 binary image일 때 잘 추출하더라
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list, cursor_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list, cursor_list

    def stave2measure(self, stave, cursor):
        # 냅다 일정 width로 나누기엔 크기 차이가 나니까 담겨있는 정보 차이도 날 거임.
        # stave를 일정 크기로 resize하기 -> height을 맞추기
        h, w = stave.shape
        max_h = self.args.max_height
        max_w = self.args.max_width

        cursor_x, _, _, _ = cursor

        # 이미지의 가로세로 비율 계산
        ratio = max_h / float(h)
        new_width = int(w * ratio)
        resized_stave = cv2.resize(stave, (new_width, max_h))

        result = []
        result_cursor = []
        start_x = 0  # 현재 이미지의 x 시작점
        _, r_w = resized_stave.shape

        # 이미지 자르기 및 패딩
        while start_x < r_w:
            end_x = min(start_x + max_w, r_w)
            cropped_image = resized_stave[:, start_x:end_x]

            length = end_x - start_x

            # 남은 부분이 120 픽셀보다 작으면 패딩을 추가합니다.
            if length < max_w:
                padding_needed = max_w - (length)
                # 오른쪽에 패딩을 추가합니다.
                cropped_image = cv2.copyMakeBorder(
                    cropped_image,
                    0,
                    0,
                    0,
                    padding_needed,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            result.append(255 - cropped_image)
            result_cursor.append(cursor_x + end_x / float(ratio))

            start_x += max_w

        return result, result_cursor

    def preprocessing(self, score_path):
        # ------------ 전처리 ------------------
        # stave 추출
        stave_list, stave_cursor_list = self.transform_score2stave(score_path)
        measure_list = []
        measure_cursor_list = []
        for idx, stave in enumerate(stave_list):
            stave_cursor = stave_cursor_list[idx]
            # measure 추출
            measures, measure_cursors = self.stave2measure(stave, stave_cursor)
            measure_list += measures
            measure_cursor_list += measure_cursors
        x_preprocessed_list = []
        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))

        return x_preprocessed_list, measure_cursor_list

    def postprocessing(self, predict_result):
        # 함수 정의
        def process_string(s):
            # 먼저, | 문자 사이의 공백을 제거
            s = re.sub(r"\s*\|\s*", "|", s)
            # 그 외의 공백은 +로 대체
            s = re.sub(r"\s+", "+", s)
            if s[-1] == "+":
                s = s[:-1]
            return s

        result_list = []
        for res in predict_result:
            result_list.append(process_string(res))
        result = "+".join(result_list)
        print(">>>>", result)
        return result

    def make_scoredata(self, score_path):
        x_preprocessed_list, cursor_list = self.preprocessing(score_path)

        # print(x_preprocessed_list)
        datetime = Util.get_datetime()
        # self.save_stave(f"demo-test-{datetime}", x_preprocessed_list)
        print(cursor_list)

        # result = self.staff2score.model_predict(x_preprocessed_list)
        # postresult = self.postprocessing(result)
        # print(postresult)
        return x_preprocessed_list, cursor_list

    def save_png(self, path, img):
        """
        save img
        """
        cv2.imwrite(f"{path}.png", img)
        print("-- shape: ", img.shape)

    def get_json_data(self, json_path):
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        return data

    def produce_all_dataset(self):
        """
        processed-feature/multi-label/.../augment 에 있는 모든 score 를 padding stave로 변환해서 img(선택), csv로 저장
        """
        # osmd raw data 가져오기
        raw_data_list_path = f"{self.args.filepaths.raw_path.osmd}"
        title_path_list = Util.get_all_subfolders(raw_data_list_path)

        for idx, title_path in enumerate(title_path_list):

            title = Util.get_title_from_dir(title_path)

            # measure 이미지 생성 : json file에서 measure 위치 가져와서 이미지 추출
            score_path = Util.get_all_files(title_path, "png")[0]
            # measure_list, _ = Score2Measure.transform_score2measure(
            #     args, title, score_path
            # )

            img_list, img_cursor_list = self.make_scoredata(score_path)

            # annotation 생성 : xml에서 measure 단위로 정보 가져와서 annotation으로 추출 및 변환
            xml_path = Util.get_all_files(title_path, "xml")[0]
            annotation = Xml2Annotation.xml2annotation_one(xml_path)
            note_list = annotation.split("+")

            # print(">>>", annotation)
            # print(">>>", img_cursor_list)
            # print(">>>", len(annotation.split("+")))
            # print(">>>", len(img_cursor_list))

            json_path = Util.get_all_files(title_path, "json")[0]
            json_data = self.get_json_data(json_path)

            json_cursor_list = json_data["cursorList"]
            json_measure_list = json_data["measureList"]

            print(">>>", json_data)
            note_idx = 0
            cursor_idx = 0  # 잘린 사진 인덱스
            result_annotation = []
            PAD = 10

            len_total_elements = sum(len(row) for row in json_cursor_list)
            len_notes = len(note_list)
            if len_total_elements == len_notes:
                # json 노트 개수와 xml 노트 개수가 맞을 경우만 통과

                # note_list : 1D
                # json_cursor_list : 2D

                for json_cursor_ in json_cursor_list:
                    annotation_tmp = []
                    while note_idx < len(json_cursor_):
                        # json_note_ : {'top': 39, 'left': 321.213, 'height': 40, 'width': 20, 'timestamp': 0.75}
                        json_note_ = json_cursor_[note_idx]
                        xml_note_ = note_list[note_idx]
                        # xml_note_: note-G5_eighth|note-C5_eighth
                        print("?>>>>>>>>", json_note_["left"])
                        print("???????>>>>>>>>", img_cursor_list[cursor_idx])
                        if json_note_["left"] + PAD < img_cursor_list[cursor_idx]:
                            annotation_tmp.append(xml_note_)
                            note_idx += 1
                        else:
                            cursor_idx += 1

                    result_annotation.append("+".join(annotation_tmp))

                # cutted_cursor = img_cursor_list[cut_idx]

                # for note_idx in range(len(note_list)):
                #     note_ = note_list[note_idx]
                #     note_ = note_list[note_idx]

            print(result_annotation)
            return

            # for idx in range(len(measure_list)):
            #     try:
            #         meas = measure_list[idx]  # measure imgs
            #         if isNeedAnno:
            #             anno = annotation_list[idx]  # annotations

            #         # measure 마다 png, txt 저장
            #         date_time = Util.get_datetime()
            #         dir_path = (
            #             f"{self.args.filepaths.feature_path.seq}/{title}/{title}-{idx}"
            #         )
            #         if isNeedAnno:
            #             os.makedirs(f"{dir_path}", exist_ok=True)
            #         file_path = f"{dir_path}/{title}_{idx}_{date_time}"

            #         print(file_path)

            #         Score2Measure.save_png(file_path, meas)
            #         if isNeedAnno:
            #             Xml2Annotation.save_txt(file_path, anno)
            #     except:
            #         print("!! -- 저장 실패")


if __name__ == "__main__":
    from configs import getconfig

    cofigpath = f"workspace/config.yaml"
    args = getconfig(cofigpath)

    # 1. 예측할 악보
    score_path = f"{args.filepaths.test_demo}/demo.png"

    handler = ProduceDataset(args)
    predict_result = handler.produce_all_dataset()
    # xml_tree = Annotation2Xml.annotation_to_musicxml(predict_result)

    # self.handler.predict(biImg_list)
