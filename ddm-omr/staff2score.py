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
char_to_int_mapping = [
    "|",  # 0
    "barline",  # 1
    "clef-percussion",  # 2
    "note-eighth",  # 3
    "note-eighth.",  # 4
    "note-half",  # 5
    "note-half.",  # 6
    "note-quarter",  # 7
    "note-quarter.",  # 8
    "note-16th",  # 9
    "note-16th.",  # 10
    "note-whole",  # 11
    "note-whole.",  # 12
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
    "timeSignature-4/4",  # 23
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
# print(char_to_num.get_vocabulary())
# test=["clef-percussion","note-eighth","|","note-eighth", "note-eighth", "note-eighth|note-eighth", "rest_whole"]
# print(char_to_num(test))
pitch_char_to_num = layers.StringLookup(
    vocabulary=list(pitch_to_int_mapping), mask_token=None
)

# 숫자를 문자로 변환
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
num_to_pitch_char = layers.StringLookup(
    vocabulary=pitch_char_to_num.get_vocabulary(), mask_token=None, invert=True
)


class StaffToScore(object):
    def __init__(self, args):
        self.args = args
        self.model = DDMOMR(args)

    def load_data_path(self, is_test=False):
        """
        각 measure에 대한 (png, txt) path 가져오기
        """
        if is_test:
            dataset_path = f"{self.args.filepaths.test_path}/"
        else:
            dataset_path = f"{self.args.filepaths.feature_path.seq}/"
        title_path_list = Util.get_all_subfolders(dataset_path)

        x_raw_path_list = []  # image
        y_raw_path_list = []  # label

        for title_path in title_path_list:
            title_dataset_path = [  # only measure folder
                item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)
            ]

            x_exp = re.compile(r"^[^._].*\.png$")  # png exp
            y_exp = re.compile(r"^[^._].*\.txt$")  # txt exp

            for tdp in title_dataset_path:
                files = os.listdir(tdp)
                x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
                y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
                if len(x_raw_path) == len(y_raw_path):  # 개수 맞는 지 확인하고 추가
                    x_raw_path_list += x_raw_path
                    y_raw_path_list += y_raw_path

        Util.print_step("load data")
        print(f"-- x: {len(x_raw_path_list)} | y: {len(y_raw_path_list)}")

        return x_raw_path_list, y_raw_path_list

    def pitch_encode_single_sample(self, img, label):
        # 1. 이미지로 변환하고 grayscale로 변환
        img = tf.expand_dims(img, axis=-1)  # 채널 추가
        # 2. [0,255]의 정수 범위를 [0,1]의 실수 범위로 변환 및 resize
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        # 3. 이미지의 가로 세로 변환
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. 라벨 값의 문자를 숫자로 변환
        label_r = pitch_char_to_num(tf.strings.split(label))

        # 7. 딕셔너리 형태로 return
        return {"image": img, "label": label_r}

    def load_data(self, is_test=False):
        x_raw_path_list, y_raw_path_list = self.load_data_path(is_test)

        x_preprocessed_list = []
        y_preprocessed_list = []

        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            # print(x_raw_path)
            y_raw_path = y_raw_path_list[idx]

            # augment 5개 데이터 생성됐다면 -> y도 그만큼 복제해주기
            x_preprocessed = self.preprocessing(x_raw_path, is_test)

            # annotation에서 띄어쓰기 있는 것들 사이사이는 + 로 연결해주기
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")

            for _, img in x_preprocessed:
                print("-- resize 후 : ", img.shape)
                x_preprocessed_list.append(img)
                y_preprocessed_list.append(y_preprocessed)

        print(
            "전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list)
        )
        result_lift, result_pitch, result_rhythm, result_note = (
            self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)
        )
        pitch_labels = result_pitch
        return x_preprocessed_list, pitch_labels

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
                self.pitch_encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        pitch_validation_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_valid, pitch_y_valid)
        )
        pitch_validation_dataset = (
            pitch_validation_dataset.map(
                self.pitch_encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # _, ax = plt.subplots(4, 1, figsize=(16, 5))
        # for batch in pitch_train_dataset.take(1):
        #     images = batch["image"]
        #     labels = batch["label"]

        #     print(">>> 데이터셋 랜덤 확인")

        #     for i in range(4):
        #         img = (images[i] * 255).numpy().astype("uint8")
        #         label = (
        #             tf.strings.reduce_join(num_to_pitch_char(labels[i]))
        #             .numpy()
        #             .decode("utf-8")
        #         )
        #         # label = labels[i]
        #         print(labels[i])
        #         ax[i].imshow(img[:, :, 0].T, cmap="gray")
        #         ax[i].set_title(label)
        #         ax[i].axis("off")
        # os.makedirs(f"dataset-output/", exist_ok=True)
        # plt.savefig(f"dataset-output/dataset.png")
        # plt.show()

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

    def predict(self):
        x, y = self.load_data(True)

        pitch_x_train, pitch_x_valid, pitch_y_train, pitch_y_valid = train_test_split(
            np.array(x), np.array(y), test_size=0.1, random_state=42
        )

        pitch_train_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_train, pitch_y_train)
        )
        pitch_train_dataset = (
            pitch_train_dataset.map(
                self.pitch_encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        pitch_validation_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_valid, pitch_y_valid)
        )
        pitch_validation_dataset = (
            pitch_validation_dataset.map(
                self.pitch_encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        model = self.model.build_model()
        checkpoint_path = f"{self.args.filepaths.checkpoints}"
        # Checkpoint 파일 경로
        check = tf.train.latest_checkpoint(checkpoint_path)
        print(">>>>>>>>>>>>>>", check)

        # 예측모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
        prediction_model.load_weights(check).expect_partial()
        prediction_model.summary()

        def pitch_decode_batch_predictions(pred):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            results = keras.backend.ctc_decode(
                pred, input_length=input_len, greedy=True
            )[0][0][:, : self.args.max_seq_len]
            output_text = []
            y_pred = []
            for res in results:
                # print(res)
                y_pred.append(res.numpy().tolist())
                res = (
                    tf.strings.reduce_join(num_to_pitch_char(res))
                    .numpy()
                    .decode("utf-8")
                    .strip("[UNK]")
                )
                output_text.append(res)
            return output_text, y_pred

        #  validation dataset에서 하나의 배치를 시각화
        for batch in pitch_validation_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            print("-- 예측 --")
            preds = prediction_model.predict(batch_images)
            pred_texts, y_pred = pitch_decode_batch_predictions(preds)

            orig_texts = []
            y_true = []
            print("-- 실제 정답 --")
            for label in batch_labels:
                # print(label)
                y_true.append(label.numpy().tolist())
                label = (
                    tf.strings.reduce_join(num_to_pitch_char(label))
                    .numpy()
                    .decode("utf-8")
                    .strip("[UNK]")
                )
                orig_texts.append(label)

            # SER 계산
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            # ser = symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            # print("Symbol Error Rate:", ser.numpy())

            # _, ax = plt.subplots(4, 4, figsize=(15, 5))
            b_l = len(pred_texts)
            _, ax = plt.subplots(b_l, 1, figsize=(80, 50))
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                title_true = f"Ground Truth: {orig_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_true}\n{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"model-predict/", exist_ok=True)
        plt.savefig(f"model-predict/pred.png")
        plt.show()

    def preprocessing(self, rgb, is_test=False):
        """
        하나의 image에 대한 전처리
        - augmentation 적용 -> binaryimage(origin), awgn,
        - model의 입력 크기에 맞게 resize & pad
        """
        augment_result = Image2Augment.process_image2augment(self.args, rgb)
        resize_result = []
        for typename, img in augment_result:
            print("-- resize 전 : ", img.shape)
            resizeimg = Image2Augment.resizeimg(self.args, img)
            if is_test:
                if typename == "origin":
                    resize_result.append((typename, resizeimg))
            else:
                resize_result.append((typename, resizeimg))

        # (확인용) 전처리 적용된 거 저장
        # for typename, img in resize_result:
        #     Image2Augment.save_augment_png("augment-output", img, typename)

        return resize_result

    # 각 token에 맞는 string list로 만들기
    def map_pitch(self, note):
        pitch_mapping = {
            "nonote": 0,
            "note-D4": 1,
            "note-E4": 2,
            "note-F4": 3,
            "note-G4": 4,
            "note-A4": 5,
            "note-B4": 6,
            "note-C5": 7,
            "note-D5": 8,
            "note-E5": 9,
            "note-F5": 10,
            "note-G5": 11,
            "note-A5": 12,
            "note-B5": 13,
            "<unk>": 14,
        }
        return "nonote" if note not in pitch_mapping else note

    def map_rhythm(self, note):
        duration_mapping = {
            "[PAD]": 0,
            "+": 1,
            "|": 2,
            "barline": 3,
            "clef-percussion": 4,
            "note-eighth": 5,
            "note-eighth.": 6,
            "note-half": 7,
            "note-half.": 8,
            "note-quarter": 9,
            "note-quarter.": 10,
            "note-16th": 11,
            "note-16th.": 12,
            "note-whole": 13,
            "note-whole.": 14,
            "rest_eighth": 15,
            "rest-eighth.": 16,
            "rest_half": 17,
            "rest_half.": 18,
            "rest_quarter": 19,
            "rest_quarter.": 20,
            "rest_16th": 21,
            "rest_16th.": 22,
            "rest_whole": 23,
            "rest_whole.": 24,
            "timeSignature-4/4": 25,
        }
        return note if note in duration_mapping else "<unk>"

    def map_lift(self, note):
        lift_mapping = {
            # "nonote"    : 0,
            "lift_null": 1,
            "lift_##": 2,
            "lift_#": 3,
            "lift_bb": 4,
            "lift_b": 5,
            "lift_N": 6,
        }
        return "nonote" if note not in lift_mapping else note

    def symbol2pitch_rhythm_lift(self, symbol_lift, symbol_pitch, symbol_rhythm):
        return (
            self.map_lift(symbol_lift),
            self.map_pitch(symbol_pitch),
            self.map_rhythm(symbol_rhythm),
        )

    def note2pitch_rhythm_lift(self, note):
        # note-G#3_eighth
        note_split = note.split("_")  # (note-G#3) (eighth)
        note_pitch_lift = note_split[:1][0]
        note_rhythm = note_split[1:][0]
        rhythm = f"note-{note_rhythm}"
        # print("-- note_rhythm: ", rhythm)

        note_note, pitch_lift = note_pitch_lift.split("-")  # (note) (G#3)
        if len(pitch_lift) > 2:
            pitch = f"note-{pitch_lift[0]+pitch_lift[-1]}"  # (G3)
            lift = f"lift_{pitch_lift[1:-1]}"
        else:
            pitch = f"note-{pitch_lift}"
            lift = f"lift_null"
        # print("-- note_pitch_lift: ", pitch, lift)
        return self.symbol2pitch_rhythm_lift(lift, pitch, rhythm)

    def rest2pitch_rhythm_lift(self, rest):
        # rest-quarter
        return self.symbol2pitch_rhythm_lift("nonote", "nonote", rest)

    def map_pitch2isnote(self, pitch_note):
        group_notes = []
        note_split = pitch_note.split("+")
        for note_s in note_split:
            if "nonote" in note_s:
                group_notes.append("nonote")
            elif "note-" in note_s:
                group_notes.append("note")
        return "+".join(group_notes)

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

            # 우선 +로 나누고, 안에 | 있는 지 확인해서 먼저 붙이기
            # note-G#3_eighth + note-G3_eighth + note-G#3_eighth|note-G#3_eighth + rest-quarter
            note_split = notes.split("+")
            for note_s in note_split:
                if "|" in note_s:
                    mapped_lift_chord = []
                    mapped_pitch_chord = []
                    mapped_rhythm_chord = []

                    # note-G#3_eighth|note-G#3_eighth
                    note_split_chord = note_s.split(
                        "|"
                    )  # (note-G#3_eighth) (note-G#3_eighth)
                    for idx, note_s_c in enumerate(note_split_chord):
                        chord_lift, chord_pitch, chord_rhythm = (
                            self.note2pitch_rhythm_lift(note_s_c)
                        )

                        mapped_lift_chord.append(chord_lift)
                        mapped_pitch_chord.append(chord_pitch)
                        mapped_rhythm_chord.append(chord_rhythm)

                        # --> '|' 도 token이기 때문에 lift, pitch엔 nonote 추가해주기
                        if idx != len(note_split_chord) - 1:
                            mapped_lift_chord.append("nonote")
                            # mapped_pitch_chord.append("nonote")

                    group_lift.append(" ".join(mapped_lift_chord))
                    group_pitch.append(" | ".join(mapped_pitch_chord))
                    group_rhythm.append(" | ".join(mapped_rhythm_chord))

                    # --> '|' 도 token이기 때문에 추가된 token 개수 더하기
                    # 동시에 친 걸 하나의 string으로 해버리는 거니까 주의하기
                    group_notes_token_len += (
                        len(note_split_chord) + len(note_split_chord) - 1
                    )

                elif "note" in note_s:
                    if "_" in note_s:
                        # note-G#3_eighth
                        note2lift, note2pitch, note2rhythm = (
                            self.note2pitch_rhythm_lift(note_s)
                        )
                        group_lift.append(note2lift)
                        group_pitch.append(note2pitch)
                        group_rhythm.append(note2rhythm)
                        group_notes_token_len += 1

                elif "rest" in note_s:
                    if "_" in note_s:
                        # rest_quarter
                        rest2lift, rest2pitch, rest2rhythm = (
                            self.rest2pitch_rhythm_lift(note_s)
                        )
                        group_lift.append(rest2lift)
                        group_pitch.append(rest2pitch)
                        group_rhythm.append(rest2rhythm)
                        group_notes_token_len += 1
                else:
                    # clef-F4+keySignature-AM+timeSignature-12/8
                    symbol2lift, symbol2pitch, symbol2rhythm = (
                        self.symbol2pitch_rhythm_lift("nonote", "nonote", note_s)
                    )
                    group_lift.append(symbol2lift)
                    group_pitch.append(symbol2pitch)
                    group_rhythm.append(symbol2rhythm)
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

            # 뒤에 남은 건 패딩
            if toks_len < self.args.max_seq_len:
                for _ in range(self.args.max_seq_len - toks_len):
                    emb_lift += " [PAD]"
                    emb_pitch += " [PAD]"
                    emb_rhythm += " [PAD]"

            result_lift.append(emb_lift)
            result_pitch.append(emb_pitch)
            result_rhythm.append(emb_rhythm)
            result_note.append(self.map_pitch2isnote(emb_pitch))
        return result_lift, result_pitch, result_rhythm, result_note
