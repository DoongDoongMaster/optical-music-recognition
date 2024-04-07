import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


from constant.common import (
    CSV,
    CURSOR,
    EXP,
    JSON,
    LABELED_FEATURE,
    MULTI_LABEL,
    NOTE_PAD,
    PNG,
    STAVE,
    STAVE_HEIGHT,
    STAVE_WIDTH,
    XML,
)
from constant.note import (
    PITCH_NOTE2CODE,
    PITCH_NOTES,
    PTICH_HEIGHT,
    REST_EIGHTH,
    REST_HALF,
    REST_QUARTER,
    REST_WHOLE,
    REST_16th,
)
from constant.path import DATA_FEATURE_PATH, DATA_RAW_PATH, OSMD
from util import Util


class FeatureLabeling:
    @staticmethod
    def process_all_feature2label():
        """
        processed-feature/ 에 있는 모든 faeture 를 label 더해 새로운 feature-labeled csv 저장
        """
        # processed-feature/multi-label 에서 title들 가져오기
        feature_path_ = f"{DATA_FEATURE_PATH}/{MULTI_LABEL}"
        feature_path_list = Util.get_all_subfolders(feature_path_)

        # title 마다 score2stave -- score가 한 장이 아니라 여러 장일 수 있으니까 반복문으로 처리
        for feature_path in feature_path_list:
            title = Util.get_title_from_dir(feature_path)
            # XML, json 파일 경로
            file_parent = f"{DATA_RAW_PATH}/{OSMD}/{title}"

            # 라벨 가져오기
            pitch_list = FeatureLabeling.process_xml2label(file_parent)

            # feature 가져오기
            csv_file_path = Util.get_all_files(f"{feature_path}", EXP[CSV])
            # label_feature 없애기
            filtered_list = [s for s in csv_file_path if LABELED_FEATURE not in s]
            feature_df = Util.load_feature_from_csv(filtered_list[0])

            # score에 pitch, width 그리기
            # json
            json_path = f"{file_parent}/{title}.{EXP[JSON]}"
            score_path = Util.get_all_files(f"{file_parent}", EXP[PNG])
            try:
                score_path = score_path[0]

                if score_path != None:
                    FeatureLabeling.draw_cursor_on_score(
                        title, MULTI_LABEL, score_path, json_path
                    )
                    FeatureLabeling.draw_label_on_cursor(
                        title, MULTI_LABEL, score_path, json_path, pitch_list
                    )

                FeatureLabeling.process_feature2label(
                    title, json_path, feature_df, pitch_list
                )
            except:
                print("해당하는 png가 없습니다.!!", score_path)

    @staticmethod
    def process_feature2label(title, json_path, feature_df, pitch_list):
        """
        1. 먼저 가로로 label df 생성 후
        2. feature df + label df.T -> new csv
        """
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        # shape: PTICH_HEIGHT x feature concat width(40700)
        label_df = pd.DataFrame(
            0, index=range(PTICH_HEIGHT), columns=range(feature_df.shape[0])
        )
        score_leftpad = data["measureList"][0][0][
            "left"
        ]  # -- stave는 score의 양옆 padding을 자르게 되니, 실제 cursor size와 달라짐. -> 맨 처음 마디의 x 만큼 sliding

        cursor_list = 0
        # cursorList-2d: row 마디 x col 노트
        for i, cursor in enumerate(data["cursorList"]):
            print("len: ", len(cursor))

            for j, point in enumerate(cursor):
                # print("row: ", i, ", col: ", j)

                top, left, height, width = FeatureLabeling.get_cursor_data(point)
                left += i * STAVE_WIDTH - score_leftpad
                # print(i, " : ", left)

                # -- 노트 인식된 곳에, xml에서 뽑아온 걸 매핑
                pitch_code = [0] * PTICH_HEIGHT
                for pitch in pitch_list[cursor_list]:
                    pitch_idx = PITCH_NOTE2CODE[pitch]
                    pitch_code[pitch_idx] = 1

                # print(pitch_code)

                right_idx = min(
                    left + width, label_df.shape[1]
                )  # -- shape을 넘길 수 있어서

                tmp_width = right_idx - left - 2 * NOTE_PAD
                pitch_code_df = [pitch_code.copy() for _ in range(tmp_width)]
                transpose_data = np.transpose(pitch_code_df)

                # label_df.loc[:, left+7: left+width-1-9 ] = transpose_data
                label_df.loc[:, left + NOTE_PAD : left + width - 1 - NOTE_PAD] = (
                    transpose_data
                )
                cursor_list += 1
                # print(label_df.loc[:, left: left + tmp_width-1])
            print("----------------------------")

        print("pitch_list len: ", len(pitch_list), "cursor len:", cursor_list)

        label_df = np.transpose(label_df)

        # 각 열에 이름 붙이기
        label_cols, feature_cols = FeatureLabeling.get_feature_label_df_column()
        label_df.columns = label_cols
        feature_df.columns = feature_cols

        merged_df = pd.concat([label_df, feature_df], axis=1)
        Util.save_feature_csv(title, merged_df, MULTI_LABEL, LABELED_FEATURE)

        # print(label_df)

    @staticmethod
    def process_xml2label(file_parent):
        """
        미리 뽑아놓은 feature csv로부터 label을 더해 새로운 feature-labeled csv 저장
        """

        # -- 1개만 존재할 테니.. 첫 번째꺼 가져오기
        xml_file_path = Util.get_all_files(f"{file_parent}", EXP[XML])

        try:
            xml_file_path = xml_file_path[0]
            pitch_list = FeatureLabeling.extract_pitch(xml_file_path)
            # # 결과를 출력합니다.
            # for i, pitches in enumerate(pitch_list, 1):
            #     print(f"Note {i}: {' '.join(pitches)}")
            return pitch_list
        except:
            print("해당하는 XML파일이 없습니다...!!", xml_file_path)

    # feature에 label 매핑
    @staticmethod
    def load_xml_data(file_path: str):
        """
        xml data 불러오기
        """
        try:
            tree = ET.parse(file_path)  # XML 파일을 파싱
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            print(f"XML 파일을 파싱하는 동안 오류가 발생했습니다: {e}")
            return None

    @staticmethod
    def extract_pitch(xml_file):
        """
        1. multiple pitch 추출
        <chord/> <-  얘 있으면 동시에 친 거임
        <unpitched>
            <display-step>A</display-step>
            <display-octave>5</display-octave>
        </unpitched>

        2. !!!!!!!!!!!!!예외!!!!!!!!!!!!!
        - grace note 제외

        3. 쉼표 추출
        <note>
            <rest/>
            <duration>48</duration>
            <type>quarter</type>
        </note>

        output : [['G5'], ['G5'], ['G5'], ['C5'], ['C5'], ['F4', 'A5'], ...]
        """

        def extract_step_octave(pitch_element):
            """
            step, octave 추출
            <unpitched>
                <step>C</step>
                <octave>5</octave>
            </unpitched>
            """
            step = pitch_element.find("display-step").text
            octave = pitch_element.find("display-octave").text
            return step, octave

        # XML 파일 파싱
        root = FeatureLabeling.load_xml_data(xml_file)

        pitch_list = []
        chord_list = []

        # 모든 <note> 엘리먼트를 찾습니다.
        for note in root.iter("note"):
            # <grace> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_grace = note.find("grace") is not None
            if is_grace:
                print("grace!")
                continue

            # <rest> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_rest = note.find("rest") is not None
            if is_rest:
                rest_element = note.find("type").text
                if rest_element == "quarter":
                    pitch_list.append([REST_QUARTER])
                elif rest_element == "eighth":
                    pitch_list.append([REST_EIGHTH])
                elif rest_element == "half":
                    pitch_list.append([REST_HALF])
                elif rest_element == "whole":
                    pitch_list.append([REST_WHOLE])
                elif rest_element == "16th":
                    pitch_list.append([REST_16th])
                continue

            pitch_elements = note.findall("./unpitched")
            # <chord> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_chord = note.find("chord") is not None
            # 만약 <chord> 엘리먼트를 가진 <note> 엘리먼트라면, 계속 추가
            if is_chord:
                for pitch_element in pitch_elements:
                    step, octave = extract_step_octave(pitch_element)
                    chord_list.append(step + octave)
            else:
                for pitch_element in pitch_elements:
                    step, octave = extract_step_octave(pitch_element)
                    chord_list = []  # -- 초기화
                    chord_list.append(step + octave)
                    pitch_list.append(chord_list)

        return pitch_list

    @staticmethod
    def get_cursor_data(point):
        """
        cursor 확인
        """
        # cursor 정보는 1024 기준이라서 x2
        top = int(point["top"])
        left = int(point["left"])
        height = int(point["height"])
        width = int(point["width"])

        return top, left, height, width

    @staticmethod
    def draw_cursor_on_score(title, label_type, image_path, json_path):
        """
        OSMD로 추출한 cursor 위치값을 score에 그려보기
        """
        # 이미지 읽어오기
        image = cv2.imread(image_path)

        # JSON 파일 읽어오기
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        # 빨간색 네모 그리기
        for cursor in data["cursorList"]:
            for point in cursor:
                top, left, height, width = FeatureLabeling.get_cursor_data(point)
                cv2.rectangle(
                    image,
                    (left + NOTE_PAD, top),
                    (left + width - NOTE_PAD, top + height),
                    (0, 0, 255),
                    2,
                )
        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}-{CURSOR}-{date_time}.{EXP[PNG]}",
            image,
        )

    @staticmethod
    def draw_label_on_cursor(title, label_type, image_path, json_path, pitch_list):
        """
        OSMD로 추출한 label을 cursor에 그려보기
        """
        # 이미지 읽어오기
        image = cv2.imread(image_path)

        # JSON 파일 읽어오기
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        print("===========================")
        print("data[cursorList] : ", len(data["cursorList"]))
        print("pitch_list: ", len(pitch_list))

        # -- data["cursorList"] 는 2D로 row: stave, col: cursor
        # -- pitch_list 는 1D로 cursor 쭉 나열되어 있음.

        cursor_idx = 0
        # cursorList-2d: row 마디 x col 노트
        for _, cursor in enumerate(data["cursorList"]):
            for _, point in enumerate(cursor):
                top, left, _, _ = FeatureLabeling.get_cursor_data(point)

                # print("!!----------- pitch_list -------------!!", cursor_idx)
                # print(len(pitch_list))

                # print(pitch_list[cursor_idx])
                # -- 노트 인식된 곳에, xml에서 뽑아온 걸 매핑
                # if len(pitch_list) <= cursor_idx:
                #     print("으엥: ", len(pitch_list), cursor_idx)
                #     continue
                for idx, pitch in enumerate(pitch_list[cursor_idx]):
                    cv2.putText(
                        image,
                        pitch,
                        (left + NOTE_PAD, top + NOTE_PAD + idx * 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cursor_idx += 1

        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}-label-{date_time}.png",
            image,
        )

    @staticmethod
    def get_feature_label_df_column() -> pd.DataFrame:
        """
        -- dataframe 헤더 초기화
        -- [G5, A5, ..., REST,...]
        """
        label_cols = PITCH_NOTES
        feature_cols = [f"{STAVE}-{i + 1}" for i in range(STAVE_HEIGHT)]

        # label + feature
        return label_cols, feature_cols

    @staticmethod
    def load_all_labeled_feature_file():
        """
        1. 모든 processed-feature에 있는 labeled-feature.csv 파일들 가져오기
        2. csv 파일들 이어 붙이기
        """
        # osmd title paths 가져오기
        title_path_ = f"{DATA_FEATURE_PATH}/{MULTI_LABEL}"
        title_path_list = Util.get_all_subfolders(title_path_)

        labeled_feature_file_list = []
        for title_path in title_path_list:
            files = Util.get_all_files(f"{title_path}", EXP[CSV])
            for file in files:
                if LABELED_FEATURE in file:
                    labeled_feature_file_list.append(file)

        # dataframe으로 합치기
        combined_df = pd.DataFrame()
        for labeled_feature in labeled_feature_file_list:
            feature_file = pd.read_csv(labeled_feature)
            combined_df = pd.concat([combined_df, feature_file], ignore_index=True)
            del feature_file

        print(
            "-- ! 로딩 완료 ! --",
            "data shape:",
            combined_df.shape,
        )
        print("-- ! features ! -- ")
        print(combined_df)

        return combined_df
