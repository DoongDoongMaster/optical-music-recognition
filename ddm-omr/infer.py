from configs import getconfig
from sheet2score import SheetToScore

def inference(score):    
    cofigpath = f"workspace/config.yaml"
    args = getconfig(cofigpath)

    handler = SheetToScore(args)
    predict_result = handler.inferSheetToXml(score)
    return predict_result