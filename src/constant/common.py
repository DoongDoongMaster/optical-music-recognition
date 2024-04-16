STAVE_HEIGHT = 120
STAVE_WIDTH = 1000

# -- note좌우 pad 크기 [ex. note width 20 중에, 8 pad * 2 -> 4px]
NOTE_PAD = 2

# -- chunk length - model에 넣기 전 dataset 가공 시
CHUNK_TIME_LENGTH = 20

PREDICT_STD = 0.5

# -- extension
PNG = "PNG"
XML = "XML"
JSON = "JSON"
CSV = "CSV"
EXP = {PNG: "png", XML: "xml", JSON: "json", CSV: "csv"}

STAVE = "stave"
PAD_STAVE = "pad-stave"
FEATURE = "feature"
LABELED_FEATURE = "labeled-feature"
CURSOR = "cursor"

OMR = "omr-seq2seq"

PITCH = "pitch"
NOTE = "note"
MULTI_CLASS = "multi-class"
MULTI_LABEL = "multi-label"

AUGMENT = "augment"
