from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

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


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # 모델이 training 시, self.add_loss()를 사용하여 loss를 계산하고 더해줌
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # test시에는 예측값만 반환
        return y_pred


class DDMOMR:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def build_model(self):
        # Inputs 정의
        input_img = layers.Input(
            shape=(self.args.max_width, self.args.max_height, 1),
            name="image",
            dtype="float32",
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # 첫번째 convolution block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # 두번째 convolution block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # 앞에 2개의 convolution block에서 maxpooling(2,2)을 총 2번 사용
        # feature map의 크기는 1/4로 downsampling 됨
        # 마지막 레이어의 filter 수는 64개 다음 RNN에 넣기 전에 reshape 해줌
        new_shape = ((self.args.max_width // 4), (self.args.max_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(
            x
        )
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(
            x
        )

        # Output layer
        x = layers.Dense(
            len(pitch_char_to_num.get_vocabulary()) + 1,
            activation="softmax",
            name="dense2",
        )(x)

        # 위에서 지정한 CTCLayer 클래스를 이용해서 ctc loss를 계산
        output = CTCLayer(name="ctc_loss")(labels, x)

        # 모델 정의
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="omr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()

        # 모델 컴파일
        model.compile(optimizer=opt)
        return model
