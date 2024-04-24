import cv2
from constant.common import AUGMENT, EXP, MULTI_LABEL, NOTE, PNG, XML
from constant.path import DATA_TEST_PATH, IMAGE_PATH, RESULT_XML_PATH
from model.multilabel_note_model import MultiLabelNoteModel
from feature_labeling import FeatureLabeling
from data_processing import DataProcessing
from Image2augment import Image2Augment
from score2stave import Score2Stave
from note2xml import Note2XML
from util import Util


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


# ======================== image augmentation ===========================
def process_image2augment():
    title = "Rock-ver"
    # Example usage
    input_image = Score2Stave.transform_img2binaryImg(f"{title}.png")
    input_image = 255 - input_image

    # Set parameters
    et_small_alpha_training = 8
    et_small_sigma_evaluation = 4

    et_large_alpha_training = 1000
    et_large_sigma_evaluation = 80

    # Apply augmentations
    awgn_image = Image2Augment.apply_awgn(input_image)

    # apn_image = Image2Augment.apply_apn(input_image)

    et_small_image = Image2Augment.apply_elastic_transform(
        input_image, et_small_alpha_training, et_small_sigma_evaluation
    )
    et_large_image = Image2Augment.apply_elastic_transform(
        input_image, et_large_alpha_training, et_large_sigma_evaluation
    )
    all_augmentations_image = Image2Augment.apply_all_augmentations(
        input_image,
        et_small_alpha_training,
        et_small_sigma_evaluation,
        et_large_alpha_training,
        et_large_sigma_evaluation,
    )

    result = [
        ("awgn_image", awgn_image),
        # ("apn_image", apn_image),
        ("et_small_image", et_small_image),
        ("et_large_image", et_large_image),
        ("all_augmentations_image", all_augmentations_image),
    ]

    for name, img in result:
        date_time = Util.get_datetime()
        output_path = f"{IMAGE_PATH}/{AUGMENT}/{title}-{name}-{date_time}.png"
        cv2.imwrite(output_path, img)


# process_image2augment()
