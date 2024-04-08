import os
import glob
import numpy as np
import pandas as pd

from constant.common import CHUNK_TIME_LENGTH, MULTI_LABEL, OMR, STAVE_HEIGHT
from constant.note import (
    CODE2NOTES,
    DURATIONS,
    DURATIONS2TYPE,
    NOTES,
    NOTES_HEIGHT,
    PITCHS,
    REST2DURATION,
    REST_NOTES,
)
from constant.path import MODEL_PATH
from feature_labeling import FeatureLabeling
from score2stave import Score2Stave
from show_result import ShowResult
from util import Util

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Input,
    Bidirectional,
    Conv2D,
    MaxPooling2D,
    Reshape,
    BatchNormalization,
    Dropout,
)


class MultiLabelNoteModel:
    def __init__(
        self,
        training_epochs,
        opt_learning_rate,
        batch_size,
        label_type,
        result_type,
        compile_mode=True,
    ):
        self.model = None
        self.training_epochs = training_epochs
        self.opt_learning_rate = opt_learning_rate
        self.batch_size = batch_size
        self.label_type = label_type
        self.result_type = result_type
        self.compile_mode = compile_mode
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_val: np.ndarray = None
        self.y_val: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.save_folder_path = f"{MODEL_PATH}/{result_type}/{label_type}"
        self.save_path = f"{self.save_folder_path}/{OMR}_{label_type}_{result_type}"
        self.model_save_type = "h5"

        self.n_rows = CHUNK_TIME_LENGTH
        self.n_columns = STAVE_HEIGHT
        self.n_classes = NOTES_HEIGHT
        self.opt_learning_rate = 0.01

    def create_model(self):
        input_layer = Input(shape=(self.n_rows, self.n_columns, 1))

        # First Convolutional Block
        conv1_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(input_layer)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        pool1 = MaxPooling2D(pool_size=(1, 3))(conv1_2)
        dropout1 = Dropout(0.2)(pool1)

        # Reshape for RNN
        reshape = Reshape((dropout1.shape[1], dropout1.shape[2] * dropout1.shape[3]))(
            dropout1
        )

        # BiGRU layers
        lstm1 = Bidirectional(LSTM(50, return_sequences=True))(reshape)
        lstm2 = Bidirectional(LSTM(50, return_sequences=True))(lstm1)
        lstm3 = Bidirectional(LSTM(50, return_sequences=True))(lstm2)
        dropout4 = Dropout(0.2)(lstm3)

        # Output layer
        output_layer = Dense(self.n_classes, activation="sigmoid")(dropout4)

        # Model compilation
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        opt = Adam(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"]
        )

    def save_model(self):
        """
        -- 학습한 모델 저장하기
        """
        date_time = Util.get_datetime()
        os.makedirs(self.save_folder_path, exist_ok=True)
        model_path = f"{self.save_path}_{date_time}.{self.model_save_type}"
        self.model.save(model_path)
        print("--! save model: ", model_path)

    def load_model(self, model_file=None):
        """
        -- method_type과 feature type에 맞는 가장 최근 모델 불러오기
        """
        model_files = glob.glob(f"{self.save_path}_*.{self.model_save_type}")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        load_model_file = model_files[0]  # 가장 최근 모델

        if model_file is not None:  # 불러오고자 하는 특정 모델 파일이 있다면
            load_model_file = model_file

        print("-- ! load model: ", load_model_file)
        self.model = tf.keras.models.load_model(
            load_model_file, compile=self.compile_mode
        )

    def create_dataset(self):
        combined_df = FeatureLabeling.load_all_labeled_feature_file()
        print("-------------------------")
        feature_df, label_df = MultiLabelNoteModel.get_x_y(MULTI_LABEL, combined_df)

        X = MultiLabelNoteModel.split_x_data(feature_df, self.n_rows)
        y = MultiLabelNoteModel.split_data(label_df, self.n_rows)

        # -- split train, val, test
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
            x_train_temp,
            y_train_temp,
            test_size=0.2,
            random_state=42,
        )
        del x_train_temp
        del y_train_temp

        self.x_train = x_train_final
        self.x_val = x_val_final
        self.x_test = x_test
        self.y_train = y_train_final
        self.y_val = y_val_final
        self.y_test = y_test

        print("x_train : ", self.x_train.shape)
        print("y_train : ", self.y_train.shape)
        print("x_val : ", self.x_val.shape)
        print("y_val : ", self.y_val.shape)
        print("x_test : ", self.x_test.shape)
        print("y_test : ", self.y_test.shape)

    def train(self):
        # Implement model train logic
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, mode="auto"
        )

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.training_epochs,
            callbacks=[early_stopping],
        )

        stopped_epoch = early_stopping.stopped_epoch
        print("--! finish train : stopped_epoch >> ", stopped_epoch, " !--")

        return history

    def evaluate(self):
        print("\n# Evaluate on test data")

        results = self.model.evaluate(
            self.x_test, self.y_test, batch_size=self.batch_size
        )
        print("test loss:", results[0])
        print("test accuracy:", results[1])

    def predict_score(self, stave_path):
        # -- resized image -> binary image
        biImg = Score2Stave.transform_img2binaryImg(stave_path)
        biImg = 255 - biImg
        biImg = np.transpose(biImg)

        # save_feature_csv("asdf", biImg, FEATURE) # -- 잘 됐는지 눈으로 확인하기 위함..
        feature = MultiLabelNoteModel.split_x_data(biImg, self.n_rows)

        predict_data = self.model.predict(feature)
        predict_data = predict_data.reshape((-1, self.n_classes))
        # -- threshold 0.5
        threshold_data = ShowResult.get_predict2threshold(predict_data, self.n_classes)
        result_dict = Util.transform_arr2dict(threshold_data)
        # ShowResult.show_label_dict_plot(result_dict)

        # 악보로 시각화
        sheet_music = ShowResult.convert_to_sheet_music(result_dict)
        ShowResult.plot_sheet_music(sheet_music, stave_path)

        # -- sequence data를 note data로 변환
        note_data = MultiLabelNoteModel.transform_seqdata2notedata(threshold_data)
        return note_data

    @staticmethod
    def get_x_y(label_type: str, feature_df: pd.DataFrame):
        if label_type in MULTI_LABEL:
            X = feature_df.drop(NOTES, axis=1).to_numpy()
            y = feature_df[NOTES].to_numpy()
            return X, y

    @staticmethod
    def split_x_data(data, chunk_size):
        num_samples, num_features = data.shape
        num_chunks = num_samples // chunk_size

        data = data[: num_chunks * chunk_size, :]
        return data.reshape((num_chunks, chunk_size, num_features, 1))

    @staticmethod
    def split_data(data, chunk_size):
        num_samples, num_features = data.shape
        num_chunks = num_samples // chunk_size

        data = data[: num_chunks * chunk_size, :]
        return data.reshape((num_chunks, chunk_size, num_features))

    @staticmethod
    def transform_seqdata2notedata(seqdata):
        # predict결과는 arr [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, ..., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],...]
        # 이를 하나의 note로 변환해주는 코드

        DIVISION = 32

        notes_result = []
        flag = False
        for row in seqdata:
            note_data = list(filter(lambda x: row[x] == 1.0, range(len(row))))
            temp_pitch_list = []
            temp_duration_list = []
            temp_rest_list = []
            for note in note_data:
                # pitch 중에 하나라도 나옴. CODE2NOTES 0 ~ 9
                # duration 중에 하나라도 나옴. 10 ~ 18
                # rest 중에 하나라도 나옴. 19 ~ 23

                if CODE2NOTES[note] in PITCHS:
                    pitch = CODE2NOTES[note]
                    temp_pitch_list.append(pitch)
                    continue
                if CODE2NOTES[note] in DURATIONS:
                    duration = CODE2NOTES[note]
                    temp_duration_list.append(duration)
                    continue
                if CODE2NOTES[note] in REST_NOTES:
                    rest = CODE2NOTES[note]
                    temp_rest_list.append(rest)
                    continue

            if len(temp_pitch_list) > 0 and len(temp_duration_list) > 0:
                if flag is True:
                    continue
                flag = True
                temp_duration = temp_duration_list[0]
                for temp_pitch_idx, temp_pitch in enumerate(temp_pitch_list):
                    temp_note = {
                        "step": temp_pitch[0],
                        "octave": int(temp_pitch[1:]),
                        "duration": int(float(temp_duration) * DIVISION),
                        "type": DURATIONS2TYPE[temp_duration],
                    }
                    if temp_pitch_idx > 0:  # 동시에 친 경우, chord 추가
                        temp_note.update({"chord": True})
                    notes_result.append(temp_note)
                continue
            if len(temp_rest_list) > 0:
                if flag is True:
                    continue
                flag = True
                temp_rest = temp_rest_list[0]
                temp_duration = REST2DURATION[temp_rest]
                temp_note = {
                    "duration": int(float(temp_duration) * DIVISION),
                    "type": DURATIONS2TYPE[temp_duration],
                }
                notes_result.append(temp_note)
                continue

            flag = False

        new_note_data = {
            "attributes": {"divisions": DIVISION, "beats": 4, "beat-type": 4},
            "notes": notes_result,
        }
        return new_note_data
