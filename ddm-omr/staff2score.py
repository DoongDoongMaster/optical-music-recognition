import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import glob
import re

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append("/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr")
from process_data.image2augment import Image2Augment
from util import Util

from sklearn.model_selection import train_test_split
from model.ddm_omr_arch import DDMOMR

# from model import TrOMR
# char_to_int_mapping = [
#     "|",  # 0
#     "barline",  # 1
#     "clef-percussion",  # 2
#     "note-eighth",  # 3
#     "note-eighth.",  # 4
#     "note-half",  # 5
#     "note-half.",  # 6
#     "note-quarter",  # 7
#     "note-quarter.",  # 8
#     "note-16th",  # 9
#     "note-16th.",  # 10
#     "note-whole",  # 11
#     "note-whole.",  # 12
#     "rest_eighth",  # 13
#     "rest_eighth.",  # 14
#     "rest_half",  # 15
#     "rest_half.",  # 16
#     "rest_quarter",  # 17
#     "rest_quarter.",  # 18
#     "rest_16th",  # 19
#     "rest_16th.",  # 20
#     "rest_whole",  # 21
#     "rest_whole.",  # 22
#     "timeSignature-4/4",  # 23
# ]


char_to_int_mapping = [
    "|",  # 0
    "barline",  # 1
    "clef-percussion",  # 2
    "timeSignature-4/4",  # 3
    # 1
    "note-D4_eighth",
    "note-D4_eighth.",
    "note-D4_half",
    "note-D4_half.",
    "note-D4_quarter",
    "note-D4_quarter.",
    "note-D4_16th",
    "note-D4_16th.",
    "note-D4_whole",
    "note-D4_whole.",
    # 2
    "note-E4_eighth",
    "note-E4_eighth.",
    "note-E4_half",
    "note-E4_half.",
    "note-E4_quarter",
    "note-E4_quarter.",
    "note-E4_16th",
    "note-E4_16th.",
    "note-E4_whole",
    "note-E4_whole.",
    # 3
    "note-F4_eighth",
    "note-F4_eighth.",
    "note-F4_half",
    "note-F4_half.",
    "note-F4_quarter",
    "note-F4_quarter.",
    "note-F4_16th",
    "note-F4_16th.",
    "note-F4_whole",
    "note-F4_whole.",
    # 4
    "note-G4_eighth",
    "note-G4_eighth.",
    "note-G4_half",
    "note-G4_half.",
    "note-G4_quarter",
    "note-G4_quarter.",
    "note-G4_16th",
    "note-G4_16th.",
    "note-G4_whole",
    "note-G4_whole.",
    # 5
    "note-A4_eighth",
    "note-A4_eighth.",
    "note-A4_half",
    "note-A4_half.",
    "note-A4_quarter",
    "note-A4_quarter.",
    "note-A4_16th",
    "note-A4_16th.",
    "note-A4_whole",
    "note-A4_whole.",
    # 6
    "note-B4_eighth",
    "note-B4_eighth.",
    "note-B4_half",
    "note-B4_half.",
    "note-B4_quarter",
    "note-B4_quarter.",
    "note-B4_16th",
    "note-B4_16th.",
    "note-B4_whole",
    "note-B4_whole.",
    # 7
    "note-C5_eighth",
    "note-C5_eighth.",
    "note-C5_half",
    "note-C5_half.",
    "note-C5_quarter",
    "note-C5_quarter.",
    "note-C5_16th",
    "note-C5_16th.",
    "note-C5_whole",
    "note-C5_whole.",
    # 8
    "note-D5_eighth",
    "note-D5_eighth.",
    "note-D5_half",
    "note-D5_half.",
    "note-D5_quarter",
    "note-D5_quarter.",
    "note-D5_16th",
    "note-D5_16th.",
    "note-D5_whole",
    "note-D5_whole.",
    # 9
    "note-E5_eighth",
    "note-E5_eighth.",
    "note-E5_half",
    "note-E5_half.",
    "note-E5_quarter",
    "note-E5_quarter.",
    "note-E5_16th",
    "note-E5_16th.",
    "note-E5_whole",
    "note-E5_whole.",
    # 10
    "note-F5_eighth",
    "note-F5_eighth.",
    "note-F5_half",
    "note-F5_half.",
    "note-F5_quarter",
    "note-F5_quarter.",
    "note-F5_16th",
    "note-F5_16th.",
    "note-F5_whole",
    "note-F5_whole.",
    # 11
    "note-G5_eighth",
    "note-G5_eighth.",
    "note-G5_half",
    "note-G5_half.",
    "note-G5_quarter",
    "note-G5_quarter.",
    "note-G5_16th",
    "note-G5_16th.",
    "note-G5_whole",
    "note-G5_whole.",
    # 12
    "note-A5_eighth",
    "note-A5_eighth.",
    "note-A5_half",
    "note-A5_half.",
    "note-A5_quarter",
    "note-A5_quarter.",
    "note-A5_16th",
    "note-A5_16th.",
    "note-A5_whole",
    "note-A5_whole.",
    # 13
    "note-B5_eighth",
    "note-B5_eighth.",
    "note-B5_half",
    "note-B5_half.",
    "note-B5_quarter",
    "note-B5_quarter.",
    "note-B5_16th",
    "note-B5_16th.",
    "note-B5_whole",
    "note-B5_whole.",
    #
    "rest_eighth",  # 13
    "rest_eighth.",  # 14
    "rest_half",  # 15
    "rest_half.",  # 16
    "rest_quarter",  # 17
    "rest_quarter.",  # 18
    "rest_16th",  # 19
    "rest_16th.",  # 20
    "rest_whole",  # 21
    "rest_whole.",  # 22
]


pitch_to_int_mapping = [
    "|",  # 1
    "nonote",  # 2
    "note-D4",  # 3
    "note-E4",  # 4
    "note-F4",  # 5
    "note-G4",  # 6
    "note-A4",  # 7
    "note-B4",  # 8
    "note-C5",  # 9
    "note-D5",  # 10
    "note-E5",  # 11
    "note-F5",  # 12
    "note-G5",  # 13
    "note-A5",  # 14
    "note-B5",  # 15
]


# 문자를 숫자로 변환
char_to_num = layers.StringLookup(vocabulary=list(char_to_int_mapping), mask_token=None)
# pitch_char_to_num = layers.StringLookup(
#     vocabulary=list(pitch_to_int_mapping), mask_token=None
# )
# 숫자를 문자로 변환
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
# num_to_pitch_char = layers.StringLookup(
#     vocabulary=pitch_char_to_num.get_vocabulary(), mask_token=None, invert=True
# )


class StaffToScore(object):
    def __init__(self, args):
        self.args = args
        self.model = DDMOMR(args)
        self.checkpoint_path = f"{self.args.filepaths.checkpoints}"
        self.prediction_model = self.load_prediction_model(self.checkpoint_path)

    def load_x_y(self, title_path):
        """ """
        # only measure folder
        title_dataset_path = [
            item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)
        ]

        x_exp = re.compile(r"^[^._].*\.png$")  # png exp
        y_exp = re.compile(r"^[^._].*\.txt$")  # txt exp

        x_raw_path_list = []  # image
        y_raw_path_list = []  # label

        for tdp in title_dataset_path:
            files = os.listdir(tdp)
            x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
            y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
            if len(x_raw_path) == len(y_raw_path):  # 개수 맞는 지 확인하고 추가
                x_raw_path_list += x_raw_path
                y_raw_path_list += y_raw_path

        return x_raw_path_list, y_raw_path_list

    def load_data_path(self):
        """
        각 measure에 대한 (png, txt) path 가져오기
        """
        dataset_path = f"{self.args.filepaths.feature_path.seq}/"
        title_path_list = Util.get_all_subfolders(dataset_path)

        x_raw_path_list = []  # image
        y_raw_path_list = []  # label

        for title_path in title_path_list:
            x_raw_path, y_raw_path = self.load_x_y(title_path)
            x_raw_path_list += x_raw_path
            y_raw_path_list += y_raw_path

            # title_dataset_path = [  # only measure folder
            #     item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)
            # ]

            # x_exp = re.compile(r"^[^._].*\.png$")  # png exp
            # y_exp = re.compile(r"^[^._].*\.txt$")  # txt exp

            # for tdp in title_dataset_path:
            #     files = os.listdir(tdp)
            #     x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
            #     y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
            #     if len(x_raw_path) == len(y_raw_path):  # 개수 맞는 지 확인하고 추가

        Util.print_step("load data")
        print(f"-- x: {len(x_raw_path_list)} | y: {len(y_raw_path_list)}")

        return x_raw_path_list, y_raw_path_list

    def encode_single_sample(self, img, label):
        # 1. 이미지로 변환하고 grayscale로 변환
        if img.shape.ndims == 2:
            img = tf.expand_dims(img, axis=-1)  # 채널 추가
        # 2. [0,255]의 정수 범위를 [0,1]의 실수 범위로 변환 및 resize
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        # 3. 이미지의 가로 세로 변환
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. 라벨 값의 문자를 숫자로 변환
        label_r = char_to_num(tf.strings.split(label))

        # 7. 딕셔너리 형태로 return
        return {"image": img, "label": label_r}

    def pitch_encode_x(self, img):
        # 1. 이미지로 변환하고 grayscale로 변환
        if img.shape.ndims == 2:
            img = tf.expand_dims(img, axis=-1)  # 채널 추가
        # 2. [0,255]의 정수 범위를 [0,1]의 실수 범위로 변환 및 resize
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        # 3. 이미지의 가로 세로 변환
        img = tf.transpose(img, perm=[1, 0, 2])

        return {"image": img}

    def load_data(self):
        x_raw_path_list, y_raw_path_list = self.load_data_path()

        x_preprocessed_list = []
        y_preprocessed_list = []

        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            # print(x_raw_path)
            y_raw_path = y_raw_path_list[idx]

            # augment 5개 데이터 생성됐다면 -> y도 그만큼 복제해주기
            x_preprocessed = self.preprocessing(x_raw_path)

            # annotation에서 띄어쓰기 있는 것들 사이사이는 + 로 연결해주기
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")

            for _, img in x_preprocessed:
                # print("-- resize 후 : ", img.shape)
                x_preprocessed_list.append(img)
                y_preprocessed_list.append(y_preprocessed)

        print(
            "전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list)
        )
        result_note = self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)

        print(result_note)

        pitch_labels = result_note
        return x_preprocessed_list, pitch_labels

    def load_test_data(self):
        test_path = f"{self.args.filepaths.test_path}/"
        x_raw_path_list, y_raw_path_list = self.load_x_y(test_path)

        x_preprocessed_list = []
        y_preprocessed_list = []

        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            y_raw_path = y_raw_path_list[idx]
            # print(">>>>>>>>>>>>>>>>>>>>", x_raw_path)

            biImg = Image2Augment.readimg(x_raw_path)
            biImg = 255 - biImg
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))

            # annotation에서 띄어쓰기 있는 것들 사이사이는 + 로 연결해주기
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")
            y_preprocessed_list.append(y_preprocessed)

        print(
            "전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list)
        )
        result_note = self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)
        pitch_labels = result_note
        return x_preprocessed_list, pitch_labels

    def predict(self, biImg_list):
        # 아래 과정을 거친 rgb 이미지 데이터를 여러 개 예측
        # biImg = Image2Augment.readimg(x_raw_path)
        # biImg = 255 - biImg
        # print(">>>>>>>>>>>>>>>>>>>>", biImg_list)
        x_preprocessed_list = []

        for img in biImg_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, img))
        print("전처리 후 x 개수: ", len(x_preprocessed_list))

        # 리스트를 tf.Tensor로 변환
        pitch_dataset = tf.data.Dataset.from_tensor_slices(
            np.array(x_preprocessed_list)
        )
        # tf.Tensor로부터 Dataset 생성
        # pitch_dataset = tf.data.Dataset.from_tensor_slices(x_preprocessed_tensor)
        # print(pitch_dataset)

        pitch_dataset = (
            pitch_dataset.map(self.pitch_encode_x, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        for batch in pitch_dataset.take(1):
            batch_images = batch["image"]

            print("-- 예측 --")
            preds = self.prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)

            b_l = len(pred_texts)
            _, ax = plt.subplots(b_l, 1)
            # Ensure ax is always iterable
            if b_l == 1:
                ax = [ax]
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"predict-result/", exist_ok=True)
        plt.savefig(f"predict-result/pred.png")
        plt.show()

    def training(self):
        x, y = self.load_data()

        pitch_x_train, pitch_x_valid, pitch_y_train, pitch_y_valid = train_test_split(
            np.array(x), np.array(y), test_size=0.1, random_state=42
        )

        pitch_train_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_train, pitch_y_train)
        )
        pitch_train_dataset = (
            pitch_train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        pitch_validation_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_valid, pitch_y_valid)
        )
        pitch_validation_dataset = (
            pitch_validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        _, ax = plt.subplots(4, 1)
        for batch in pitch_train_dataset.take(1):
            images = batch["image"]
            labels = batch["label"]

            print(">>> 데이터셋 랜덤 확인")

            for i in range(4):
                img = (images[i] * 255).numpy().astype("uint8")
                label = (
                    tf.strings.reduce_join(num_to_char(labels[i]))
                    .numpy()
                    .decode("utf-8")
                )
                # label = labels[i]
                # print(labels[i])
                ax[i].imshow(img[:, :, 0].T, cmap="gray")
                ax[i].set_title(label)
                ax[i].axis("off")
        os.makedirs(f"dataset-output/", exist_ok=True)
        plt.savefig(f"dataset-output/dataset-1.png")
        plt.show()

        # Get the model
        model = self.model.build_model()
        model.summary()

        epochs = self.args.epoch
        early_stopping_patience = 10
        # early stopping 지정
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
        os.makedirs(f"{self.args.filepaths.checkpoints}", exist_ok=True)

        # Define a checkpoint callback
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self.args.filepaths.checkpoints, "omr-checkpoint.ckpt"
            ),
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        # Save model weights to a .ckpt file
        model.save_weights(
            os.path.join(self.args.filepaths.checkpoints, "omr-checkpoint.ckpt")
        )
        # Train the model with checkpointing
        model.fit(
            pitch_train_dataset,
            validation_data=pitch_validation_dataset,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint_callback],
        )

        # 예측 모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output,
        )

        #  validation dataset에서 하나의 배치를 시각화
        for batch in pitch_validation_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            print("-- 예측 --")
            preds = prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)

            orig_texts = []
            y_true = []
            print("-- 실제 정답 --")
            for label in batch_labels:
                # print(label)
                y_true.append(label.numpy().tolist())
                label = (
                    tf.strings.join(num_to_char(label), separator=" ")
                    .numpy()
                    .decode("utf-8")
                    .replace("[UNK]", "")
                )
                orig_texts.append(label)

            # SER 계산
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            # ser = symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            # print("Symbol Error Rate:", ser.numpy())

            # _, ax = plt.subplots(4, 4, figsize=(15, 5))
            b_l = len(pred_texts)
            _, ax = plt.subplots(b_l, 1)
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                title_true = f"Ground Truth: {orig_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_true}\n{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"test-result/", exist_ok=True)
        plt.savefig(f"test-result/pred.png")
        plt.show()

    def load_prediction_model(self, checkpoint_path):
        model = self.model.build_model()
        check = tf.train.latest_checkpoint(checkpoint_path)
        print("-- Loading weights from:", check)

        # 예측 모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output,
        )
        prediction_model.load_weights(check).expect_partial()
        prediction_model.summary()
        return prediction_model

    def pitch_decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[
            0
        ][0][:, : self.args.max_seq_len]
        output_text = []
        y_pred = []
        for res in results:
            # print(res)
            y_pred.append(res.numpy().tolist())
            res = (
                tf.strings.join(num_to_char(res), separator=" ")
                .numpy()
                .decode("utf-8")
                .replace("[UNK]", "")
            )
            output_text.append(res)
        return output_text, y_pred

    def test(self):
        x, y = self.load_test_data()

        pitch_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        pitch_dataset = (
            pitch_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        #  validation dataset에서 하나의 배치를 시각화
        for batch in pitch_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            print("-- 예측 --")
            preds = self.prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)

            orig_texts = []
            y_true = []
            print("-- 실제 정답 --")
            for label in batch_labels:
                # print(label)
                y_true.append(label.numpy().tolist())
                label = (
                    tf.strings.join(num_to_char(label), separator=" ")
                    .numpy()
                    .decode("utf-8")
                    .replace("[UNK]", "")
                )
                orig_texts.append(label)

            # SER 계산
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            # ser = symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            # print("Symbol Error Rate:", ser.numpy())

            # _, ax = plt.subplots(4, 4, figsize=(15, 5))
            b_l = len(pred_texts)
            _, ax = plt.subplots(b_l, 1)
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                title_true = f"Ground Truth: {orig_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_true}\n{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"predict-result/", exist_ok=True)
        plt.savefig(f"predict-result/pred.png")
        plt.show()

    def preprocessing(self, rgb):
        """
        하나의 image에 대한 전처리
        - augmentation 적용 -> binaryimage(origin), awgn,
        - model의 입력 크기에 맞게 resize & pad
        """
        augment_result = Image2Augment.process_image2augment(self.args, rgb)
        resize_result = []
        for typename, img in augment_result:
            # print("-- resize 전 : ", img.shape)
            resizeimg = Image2Augment.resizeimg(self.args, img)
            resize_result.append((typename, resizeimg))

        # (확인용) 전처리 적용된 거 저장
        for typename, img in augment_result:
            Image2Augment.save_augment_png("augment-output", img, typename)

        # (확인용) 전처리 적용된 거 저장
        for typename, img in resize_result:
            Image2Augment.save_augment_png("zeropadding-output", img, typename)

        return resize_result

    def map_notes2pitch_rhythm_lift_note(self, note_list):
        result_lift = []
        result_pitch = []
        result_rhythm = []
        result_note = []

        for notes in note_list:
            group_lift = []
            group_pitch = []
            group_rhythm = []
            group_notes_token_len = 0
            group_note = []

            # 우선 +로 나누고, 안에 | 있는 지 확인해서 먼저 붙이기
            # note-G#3_eighth + note-G3_eighth + note-G#3_eighth|note-G#3_eighth + rest-quarter
            note_split = notes.split("+")
            for note_s in note_split:
                if "|" in note_s:
                    mapped_lift_chord = []
                    mapped_pitch_chord = []
                    mapped_rhythm_chord = []
                    mapped_note_chord = []

                    # note-G#3_eighth|note-G#3_eighth
                    # (note-G#3_eighth) (note-G#3_eighth)
                    note_split_chord = note_s.split("|")
                    mapped_note_chord = note_split_chord
                    # for idx, note_s_c in enumerate(note_split_chord):
                    #     chord_lift, chord_pitch, chord_rhythm = (
                    #         self.note2pitch_rhythm_lift(note_s_c)
                    #     )

                    #     mapped_lift_chord.append(chord_lift)
                    #     mapped_pitch_chord.append(chord_pitch)
                    #     mapped_rhythm_chord.append(chord_rhythm)

                    #     # --> '|' 도 token이기 때문에 lift, pitch엔 nonote 추가해주기
                    #     if idx != len(note_split_chord) - 1:
                    #         mapped_lift_chord.append("nonote")
                    #         # mapped_pitch_chord.append("nonote")

                    # group_lift.append(" ".join(mapped_lift_chord))
                    # group_pitch.append(" | ".join(mapped_pitch_chord))
                    # group_rhythm.append(" | ".join(mapped_rhythm_chord))

                    group_note.append(" | ".join(mapped_note_chord))

                    # --> '|' 도 token이기 때문에 추가된 token 개수 더하기
                    # 동시에 친 걸 하나의 string으로 해버리는 거니까 주의하기
                    group_notes_token_len += (
                        len(note_split_chord) + len(note_split_chord) - 1
                    )

                # elif "note" in note_s:
                #     if "_" in note_s:
                #         # # note-G#3_eighth
                #         # note2lift, note2pitch, note2rhythm = (
                #         #     self.note2pitch_rhythm_lift(note_s)
                #         # )
                #         # group_lift.append(note2lift)
                #         # group_pitch.append(note2pitch)
                #         # group_rhythm.append(note2rhythm)

                #         group_notes_token_len += 1

                # elif "rest" in note_s:
                #     if "_" in note_s:
                #         # rest_quarter
                #         # rest2lift, rest2pitch, rest2rhythm = (
                #         #     self.rest2pitch_rhythm_lift(note_s)
                #         # )
                #         # group_lift.append(rest2lift)
                #         # group_pitch.append(rest2pitch)
                #         # group_rhythm.append(rest2rhythm)
                #         group_notes_token_len += 1
                # else:
                #     # # clef-F4+keySignature-AM+timeSignature-12/8
                #     # symbol2lift, symbol2pitch, symbol2rhythm = (
                #     #     self.symbol2pitch_rhythm_lift("nonote", "nonote", note_s)
                #     # )
                #     # group_lift.append(symbol2lift)
                #     # group_pitch.append(symbol2pitch)
                #     # group_rhythm.append(symbol2rhythm)
                #     group_notes_token_len += 1

                else:
                    group_note.append(note_s)
                    group_notes_token_len += 1

            toks_len = group_notes_token_len

            # lift, pitch
            # emb_lift="nonote+"
            # emb_pitch="nonote+"

            emb_lift = " ".join(group_lift)
            emb_pitch = " ".join(group_pitch)
            # emb_lift+="+nonote"
            # emb_pitch+="+nonote"

            # rhythm
            # emb_rhythm="[BOS]"
            emb_rhythm = " ".join(group_rhythm)
            # emb_rhythm+="[EOS]"

            emb_note = " ".join(group_note)

            # 뒤에 남은 건 패딩
            if toks_len < self.args.max_seq_len:
                for _ in range(self.args.max_seq_len - toks_len):
                    # emb_lift += " [PAD]"
                    # emb_pitch += " [PAD]"
                    # emb_rhythm += " [PAD]"
                    emb_note += " [PAD]"

            # result_lift.append(emb_lift)
            # result_pitch.append(emb_pitch)
            # result_rhythm.append(emb_rhythm)
            # result_note.append(self.map_pitch2isnote(emb_pitch))

            result_note.append(emb_note)
        return result_note
