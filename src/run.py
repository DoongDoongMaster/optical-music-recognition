from constant.common import EXP, MULTI_LABEL, PITCH, PNG, XML
from constant.path import DATA_TEST_PATH, RESULT_XML_PATH
from model.multilabel_pitch_model import MultiLabelPitchModel
from feature_labeling import FeatureLabeling
from data_processing import DataProcessing
from note2xml import Note2XML
from util import Util


# ======================== omr ai ===========================
def process_all_data():
    # save feature csv
    DataProcessing.process_all_score2stave()

    # save labeled-feature csv
    FeatureLabeling.process_all_feature2label()


multilabel_pitch_model = MultiLabelPitchModel(40, 0.001, 32, MULTI_LABEL, PITCH)


def train_model():
    # get feature, label from csv, train
    multilabel_pitch_model.create_dataset()
    multilabel_pitch_model.create_model()
    multilabel_pitch_model.train()
    multilabel_pitch_model.evaluate()
    multilabel_pitch_model.save_model()


def predict_model():
    # predict_test_datas = [
    #     "didyouloveme_pad-stave_0_2024-04-06_16-26-04.png",
    #     "Rock-ver_pad-stave_11_2024-04-06_16-26-09.png",
    #     "uptownfunk_pad-stave_17_2024-04-06_16-26-10.png",
    #     "Rock-ver_pad-stave_9_2024-04-06_16-26-09.png",
    # ]
    # for predict_test_data in predict_test_datas:
    #     multilabel_pitch_model.predict_score(f"{DATA_TEST_PATH}/{predict_test_data}")
    files = Util.get_all_files(f"{DATA_TEST_PATH}", EXP[PNG])
    multilabel_pitch_model.load_model()
    for file_path in files:
        multilabel_pitch_model.predict_score(f"{file_path}")


# process_all_data()
train_model()
predict_model()


# ======================== note2xml ===========================
def create_musicxml_dummy():
    note_data = {
        "attributes": {"divisions": 32, "beats": 4, "beat-type": 4},
        "notes": [
            {"step": "F", "octave": 4, "duration": 32, "type": "quarter"},
            {
                "step": "G",
                "octave": 5,
                "duration": 32,
                "type": "quarter",
                "chord": True,
            },
            {"duration": 32, "type": "quarter"},
            {"step": "A", "octave": 4, "duration": 32, "type": "quarter"},
        ],
    }

    # Create ElementTree and write to file
    datetime = Util.get_datetime()
    Note2XML.create_musicxml(
        note_data, f"{RESULT_XML_PATH}/musicxml-{datetime}.{EXP[XML]}"
    )
