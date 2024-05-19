import os
import sys

sys.path.append(
    "/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr/produce_data"
)
from xml2annotation import Xml2Annotation
from score2measure import Score2Measure
from util import Util


class DataProcessing:
    @staticmethod
    def process_all_score2measure(args):
        """
        processed-feature/multi-label/.../augment 에 있는 모든 score 를 padding stave로 변환해서 img(선택), csv로 저장
        """
        # osmd raw data 가져오기
        raw_data_list_path = f"{args.filepaths.raw_path.osmd}"
        title_path_list = Util.get_all_subfolders(raw_data_list_path)

        for title_path in title_path_list:
            title = Util.get_title_from_dir(title_path)

            # measure 이미지 생성 : json file에서 measure 위치 가져와서 이미지 추출
            score_path = Util.get_all_files(title_path, "png")[0]
            measure_list, _ = Score2Measure.transform_score2measure(
                args, title, score_path
            )
            # annotation 생성 : xml에서 measure 단위로 정보 가져와서 annotation으로 추출 및 변환
            xml_path = Util.get_all_files(title_path, "xml")[0]
            annotation_list = Xml2Annotation.process_xml2annotation(xml_path)
            # annotation_list = Util.get_all_files(f"../data/processed-feature/transformer/Rock-ver/annotation/", "txt")

            if len(measure_list) != len(annotation_list):
                print("!! -- measure 와 annotation 개수가 맞지 않음.")
                continue

            for idx in range(len(measure_list)):
                try:
                    meas = measure_list[idx]  # measure imgs
                    anno = annotation_list[idx]  # annotations

                    # measure 마다 png, txt 저장
                    date_time = Util.get_datetime()
                    dir_path = f"{args.filepaths.feature_path.seq}/{title}/{title}_{args.measure}_{idx+1:02d}"
                    os.makedirs(f"{dir_path}", exist_ok=True)
                    file_path = (
                        f"{dir_path}/{title}_{args.measure}_{idx+1:02d}_{date_time}"
                    )

                    Score2Measure.save_png(file_path, meas)
                    Xml2Annotation.save_txt(file_path, anno)
                except:
                    print("!! -- 저장 실패")
