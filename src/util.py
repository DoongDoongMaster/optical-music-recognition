from datetime import datetime
import glob
import os

import pandas as pd

from constant import CODE2PITCH_NOTE, DATA_FEATURE_PATH


class Util:
    @staticmethod
    def get_datetime():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def get_title(score_path):
        return os.path.basename(os.path.dirname(score_path))

    @staticmethod
    def get_title_from_dir(score_path):
        return os.path.basename(score_path)

    @staticmethod
    def get_all_files(parent_folder_path, exp):
        all_file_list = glob.glob(f"{parent_folder_path}/*")
        file_list = [file for file in all_file_list if file.endswith(f".{exp}")]
        return file_list

    @staticmethod
    def get_all_subfolders(folder_path):
        subfolder_paths = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolder_paths.append(item_path)
        return subfolder_paths

    @staticmethod
    def load_feature_from_csv(csv_file_path):
        df = pd.read_csv(csv_file_path)
        print(f"csv shape: {df.shape}")
        return df

    @staticmethod
    def save_feature_csv(title, features, label_type, state):
        """
        state : LABELED_FEATURE | FEATURE
        """
        os.makedirs(f"{DATA_FEATURE_PATH}/{label_type}/{title}/", exist_ok=True)
        date_time = Util.get_datetime()
        save_path = (
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}_{state}_{date_time}.csv"
        )
        df = pd.DataFrame(features)
        df.to_csv(save_path, index=False)
        print(f"{title} - features shape: {df.shape}")

    @staticmethod
    def transform_arr2dict(arr_data):
        print("shape:", arr_data.shape)

        result_dict = {}
        for code, drum in CODE2PITCH_NOTE.items():
            # print("--", code, drum)

            result_dict[drum] = [row[code] for row in arr_data]
            print(result_dict[drum])
        return result_dict