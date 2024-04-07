# -- rest
REST_QUARTER = "REST_QUARTER"
REST_EIGHTH = "REST_EIGHTH"
REST_HALF = "REST_HALF"
REST_WHOLE = "REST_WHOLE"
REST_16th = "REST_16th"

REST_NOTES = [
    REST_QUARTER,
    REST_EIGHTH,
    REST_HALF,
    REST_WHOLE,
    REST_16th,
]

# -- pitch
PITCH_NOTES = [
    "D4",
    "F4",
    "A4",
    "C5",
    "D5",
    "E5",
    "F5",
    "G5",
    "A5",
    "B5",
] + REST_NOTES

PTICH_HEIGHT = len(PITCH_NOTES)
# {0: 'A3', 1: 'B3', 2: 'C4', 3: 'D4', 4: 'E4', 5: 'F4', 6: 'G4', 7: 'A4', 8: 'B4', 9: 'C5', 10: 'D5', 11: 'E5', 12: 'F5', 13: 'G5', 14: 'A5', 15: 'B5', 16: 'C6'}
CODE2PITCH_NOTE = {index: note for index, note in enumerate(PITCH_NOTES)}
# {'A3': 0, 'B3': 1, 'C4': 2, 'D4': 3, 'E4': 4, 'F4': 5, 'G4': 6, 'A4': 7, 'B4': 8, 'C5': 9, 'D5': 10, 'E5': 11, 'F5': 12, 'G5': 13, 'A5': 14, 'B5': 15, 'C6': 16}
PITCH_NOTE2CODE = {note: index for index, note in enumerate(PITCH_NOTES)}

# -- duration
# 4분 음표 기준 1
DURATION_NOTES = [
    0.250,
    0.375,
    0.500,
    0.750,
    1.000,
    1.500,
    2.000,
    3.000,
    4.000,
] + REST_NOTES
DURATION_HEIGHT = len(DURATION_NOTES)
# {0: 0.25, 1: 0.375, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.5, 6: 2.0, 7: 3.0, 8: 4.0, 9: 'REST_QUARTER', 10: 'REST_EIGHTH', 11: 'REST_HALF', 12: 'REST_WHOLE'}
CODE2DURATION_NOTE = {index: note for index, note in enumerate(DURATION_NOTES)}
# {0.25: 0, 0.375: 1, 0.5: 2, 0.75: 3, 1.0: 4, 1.5: 5, 2.0: 6, 3.0: 7, 4.0: 8, 'REST_QUARTER': 9, 'REST_EIGHTH': 10, 'REST_HALF': 11, 'REST_WHOLE': 12}
DURATION_NOTE2CODE = {note: index for index, note in enumerate(DURATION_NOTES)}
