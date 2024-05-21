import os
import re
import cv2
from util import Util
from music21 import stream, instrument, clef, meter, note, percussion
from process_data.image2augment import Image2Augment
import sys
import xml.etree.ElementTree as ET
from music21.musicxml import m21ToXml

# sys.path.append("~/srv/projects/optical-music-recognition/ddm-omr")
from staff2score import StaffToScore

# 상수들
BARLINE = "barline"
NOTE = "note"
REST = "rest"
NOTEHEAD_X_LIST = ("G5", "A5", "D4")
STEM_DIRECTION_UP = "up"
DIVISION_CHORD = "+"
DIVISION_NOTE = "|"
DIVISION_DURATION = "_"
DIVISION_PITCH = "-"

# -- annotation dict key 값들
IS_NOTE = "is_note"
NOTEHEAD = "notehead"
NOTEHEAD_X = "x"
PITCH = "pitch"
DURATION = "duration"

# -- duration type <-> quarterLength match dict
DURATION_TYPE_TO_LENGTH_TEMP = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
    "256th": 0.015625,
    "512th": 0.0078125,
    "1024th": 0.00390625,
    "breve": 8.0,
    "longa": 16.0,
    "maxima": 32.0,
}
DURATION_TYPE_TO_LENGTH = {}
for duration_type, quarter_length in DURATION_TYPE_TO_LENGTH_TEMP.items():
    DURATION_TYPE_TO_LENGTH[duration_type] = quarter_length
    DURATION_TYPE_TO_LENGTH[duration_type + "."] = quarter_length + quarter_length / 2


class SheetToScore(object):
    def __init__(self, args):
        self.args = args
        self.staff2score = StaffToScore(args)

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

    def stave2measure(self, stave):
        # 냅다 일정 width로 나누기엔 크기 차이가 나니까 담겨있는 정보 차이도 날 거임.
        # stave를 일정 크기로 resize하기 -> height을 맞추기
        h, w = stave.shape
        max_h = self.args.max_height
        max_w = self.args.max_width

        # 이미지의 가로세로 비율 계산
        new_width = int((max_h / h) * w)
        resized_stave = cv2.resize(stave, (new_width, max_h))

        result = []
        start_x = 0  # 현재 이미지의 x 시작점
        _, r_w = resized_stave.shape

        # 이미지 자르기 및 패딩
        while start_x < r_w:
            end_x = min(start_x + max_w, r_w)
            cropped_image = resized_stave[:, start_x:end_x]

            # 남은 부분이 120 픽셀보다 작으면 패딩을 추가합니다.
            if end_x - start_x < max_w:
                padding_needed = max_w - (end_x - start_x)
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

            start_x += max_w

        return result

    def predict(self, score_path):
        stave_list = self.transform_score2stave(score_path)  # stave 추출
        measure_list = []
        for idx, stave in enumerate(stave_list):
            measures = self.stave2measure(stave)  # measure 추출
            measure_list += measures

        # ------------ 전처리 ------------------
        x_preprocessed_list = []
        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))

        # 함수 정의
        def process_string(s):
            # 먼저, | 문자 사이의 공백을 제거
            s = re.sub(r"\s*\|\s*", "|", s)
            # 그 외의 공백은 +로 대체
            s = re.sub(r"\s+", "+", s)
            if s[-1] == "+":
                s = s[:-1]
            return s

        result = self.staff2score.model_predict(x_preprocessed_list)
        result_list = []
        for res in result:
            result_list.append(process_string(res))
        result = "+".join(result_list)
        print(">>>>", result)
        return result

    """
    result dict : 4/4 박자에 맞는 마디별 음표 정보

    annotation: 
    [[[{'duration': 1.0, 'is_note': True, 'pitch': 'G5', 'notehead': 'x'}],
    [{'duration': 1.0, 'is_note': False}],
    [{'duration': 1.0, 'is_note': True, 'pitch': 'G5', 'notehead': 'x'}],
    [{'duration': 0.5, 'is_note': True, 'pitch': 'C5', 'notehead': None}],
    [{'duration': 1.0, 'is_note': True, 'pitch': 'C5', 'notehead': None}]]]

    result: 
    [[[{'duration': 1.0, 'is_note': True, 'pitch': 'G5', 'notehead': 'x'}],
    [{'duration': 1.0, 'is_note': False}],
    [{'duration': 1.0, 'is_note': True, 'pitch': 'G5', 'notehead': 'x'}],
    [{'duration': 0.5, 'is_note': True, 'pitch': 'C5', 'notehead': None}],
    [{'duration': 0.5, 'is_note': False}]]]
    """

    def fit_annotation_bar(self, annotation_dict):
        result_annotation = []
        total_duration_4_4 = 4.0

        for bar in annotation_dict:
            sum_duration = 0
            chord_annotation = []
            for chord_info in bar:
                first_duration = chord_info[0][DURATION]
                if sum_duration + first_duration > total_duration_4_4:
                    break
                chord_annotation.append(chord_info)
                sum_duration += first_duration
            rest_duration = total_duration_4_4 - sum_duration
            if rest_duration > 0:
                chord_annotation.append([{DURATION: rest_duration, IS_NOTE: False}])
            result_annotation.append(chord_annotation)

        return result_annotation

    def m21_score_to_xml_tree(self, m21_score):
        # MusicXML string으로 변환
        musicxml_string = m21ToXml.GeneralObjectExporter(m21_score).parse()

        # XML Tree 구조로 변환
        xml_tree = ET.ElementTree(ET.fromstring(musicxml_string))

        # XML 파일로 저장
        # xml_tree.write('example_score.xml', encoding='utf-8', xml_declaration=True)

        return xml_tree

    """
    result dict

    [
        # 마디
        [
            # 마디 안의 동시에 친 음표
            [
                {
                    "is_note": True, 
                    "notehead": 'x', 
                    "pitch": 'G5',
                    "duration": 1.0,
                },
                {
                    "is_note": True, 
                    "notehead": None, 
                    "pitch": 'G5',
                    "duration": 0.75,
                },
                ...
            ],
            ...
        ],
        ...
    ]
    """

    def split_annotation(self, annotation):
        annotation_dict_list = []

        # 마디 기준 ('barline') 으로 자르기
        bar_list = annotation.split(BARLINE)

        for bar_info in bar_list:
            if bar_info == "":
                continue

            # 동시에 친 음표 기준 ('+') 으로  자르기
            chord_list = bar_info.split(DIVISION_CHORD)
            annotation_chord_list = []
            for chord_info in chord_list:
                if chord_info == "" or (
                    chord_info[0:4] != NOTE and chord_info[0:4] != REST
                ):
                    continue

                # 노트 얻기 ('|' 기준으로 자르기)
                note_list = chord_info.split(DIVISION_NOTE)
                annotation_note_list = []
                for note_info in note_list:
                    if note_info == "":
                        continue

                    # 노트 정보 객체
                    note_info_dict = {}

                    # note, rest, pitch, duration 얻기
                    pitch_info, duration = note_info.split(DIVISION_DURATION)
                    pitch_info_list = pitch_info.split(DIVISION_PITCH)
                    note_info_dict[DURATION] = DURATION_TYPE_TO_LENGTH[duration]
                    note_info_dict[IS_NOTE] = pitch_info_list[0] == NOTE
                    if pitch_info_list[0] == NOTE:  # 음표
                        note_info_dict[IS_NOTE] = True
                        note_info_dict[PITCH] = pitch_info_list[1]
                        note_info_dict[NOTEHEAD] = None
                        if pitch_info_list[1] in NOTEHEAD_X_LIST:
                            note_info_dict[NOTEHEAD] = NOTEHEAD_X

                    annotation_note_list.append(note_info_dict)
                annotation_chord_list.append(annotation_note_list)
            annotation_dict_list.append(annotation_chord_list)

        return annotation_dict_list

    """
    xml tree 형태 리턴
    """

    def annotation_to_musicxml(self, annotation):
        annotation_dict = self.split_annotation(annotation)
        annotation_dict = self.fit_annotation_bar(annotation_dict)

        # Score 객체 생성
        score = stream.Score()

        # Drum Track 생성
        drum_track = stream.Part()
        drum_track.append(instrument.Percussion())

        # Drum Clef 생성
        drum_clef = clef.PercussionClef()
        drum_track.append(drum_clef)

        # 4/4 Time Signature 생성
        time_signature = meter.TimeSignature("4/4")
        drum_track.append(time_signature)

        for bar in annotation_dict:
            for chord_info in bar:
                chord_notes = []
                is_note = any(
                    item[IS_NOTE] for item in chord_info
                )  # 하나라도 음표 있다면

                if not is_note:  # 쉼표
                    r = note.Rest()
                    r.duration.quarterLength = chord_info[0][DURATION]
                    drum_track.append(r)
                    continue

                # 음표
                for note_info in chord_info:
                    if note_info[IS_NOTE]:  # 음표
                        # unpitched 음표 생성
                        n = note.Unpitched(displayName=note_info[PITCH])
                        n.duration.quarterLength = note_info[DURATION]
                        n.stemDirection = STEM_DIRECTION_UP

                        if note_info[NOTEHEAD] != None:
                            n.notehead = note_info[NOTEHEAD]
                        chord_notes.append(n)

                chord = percussion.PercussionChord(chord_notes)
                chord.stemDirection = (
                    STEM_DIRECTION_UP  # Chord의 모든 노트의 꼬리 방향을 위로 설정
                )
                drum_track.append(chord)

        # Score에 Drum Track 추가
        score.insert(0, drum_track)

        score.show()

        xml_tree = self.m21_score_to_xml_tree(score)

        return xml_tree


if __name__ == "__main__":
    from configs import getconfig

    cofigpath = f"workspace/config.yaml"
    args = getconfig(cofigpath)

    # 1. 예측할 악보
    score_path = f"{args.filepaths.raw_path.osmd}/Rock-ver/Rock-ver.png"

    handler = SheetToScore(args)
    predict_result = handler.predict(score_path)
    xml_tree = handler.annotation_to_musicxml(predict_result)

    # self.handler.predict(biImg_list)
