import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gradio as gr
import torch

from core.TFTmodel import TFT_property, scan_core
# unfinished

def predict(input):
    out_text = ''
    if input == None:
        return "Please input \".csv\" or \".xlsx\" file"
    if input.name.split('.')[-1].lower() == 'xlsx':
        df = pd.read_excel(input.name, header=0)
    elif input.name.split('.')[-1].lower() == 'csv':
        df = pd.read_csv(input.name, header=0)
    else:
        return "Wrong input, please input \".csv\" or \".xlsx\" file."

    with torch.no_grad():
        prediction = torch.nn.functional
    return out_text