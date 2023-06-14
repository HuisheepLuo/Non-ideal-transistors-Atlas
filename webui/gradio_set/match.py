import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gradio as gr

from core.TFTmodel import TFT_property, scan_core
from core.process import extract_param, model_generation, normalized, load_dataframe, fig2img, AED, MSE, DTW, LCSS


matplotlib.use('Agg')
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Arial']

def param_fit_input(device_width, device_length, device_C_ox, device_thickness_sc, voltage_fix, 
                voltage_max, mu0, Vth0, scan_type, is_back_mode, is_log, is_sqrt_mode, is_normalized, algorithm, 
                semi_type, df_in, input_file_object):
    #-----------------------------------------------#
    TFT_prop = TFT_property(device_width, device_length, device_C_ox, device_thickness_sc)
    voltage_step = voltage_max / 100
    outp_text = ''
    # input file check
    if input_file_object == None:
        return "Please input \".csv\" or \".xlsx\" file", None
    if input_file_object.name.split('.')[-1].lower() == 'xlsx':
        df = pd.read_excel(input_file_object.name, header=0)
    elif input_file_object.name.split('.')[-1].lower() == 'csv':
        df = pd.read_csv(input_file_object.name, header=0)
    else:
        return "Wrong input, please input \".csv\" or \".xlsx\" file.", None

    if semi_type == 'p-type':
        seq_test = np.array([-df.iloc[:, 0], df.iloc[:, 1]])
    else:
        seq_test = np.array([df.iloc[:, 0], df.iloc[:, 1]])
    mobility, Vth, x_d = extract_param(seq_test, TFT_prop)
    outp_text = f'Mobility:{mobility:02f}, Vth:{Vth:02f}'
    print(outp_text)
    #-----------------------------------------------#

    if scan_type.lower() == 'transfer':
        transfer_mode = True
    else: transfer_mode = False

    s_core = scan_core(transfer_mode, voltage_fix, voltage_max, voltage_step,
                    property=TFT_prop, back=is_back_mode)
    
    TFT_prop.renew_param('mu10', mu0)
    TFT_prop.renew_param('mu20', mu0)
    s_core.renew_param('Vth', Vth0)

    if transfer_mode:
        x_name = 'VG'
    else:
        x_name = 'VD'
    if is_sqrt_mode:
        y_name = 'I_tot_sqrt'
    else:
        y_name = 'I_tot'

    param_group = load_dataframe(df_in)
    point_lib = model_generation(param_group, TFT_prop, s_core, x_name, y_name, is_normalized, is_comb=True)

    plt.close('all')
    counts = len(point_lib)
    imgs = []


    dis_array = np.zeros(counts) # storing distance of two sequences. 
    if is_normalized:
        seq_test = normalized(seq_test.T).T
        mobility, Vth, x_d = extract_param(seq_test, TFT_prop)
    Vth = 0
    x_t, y_t = seq_test
    #----------------input curve------------------#
    f, ax = plt.subplots(1,1, figsize=(6,6))
    f.subplots_adjust(left=0.2, right=0.95)
    ax.plot(x_t, y_t, color='red')
    imgs.append((fig2img(f), 'Input curve'))
    plt.close('all')

    #----------------evaluation function------------------#
    if algorithm == 'AED':
        e_func = AED
    elif algorithm == 'MSE':
        e_func = MSE
    elif algorithm == 'DTW':
        e_func = DTW
    elif algorithm == 'LCSS':
        e_func = LCSS

    #----------------evaluation------------------#
    if not is_back_mode:
        seq_test_part = np.array([[x_t[i], y_t[i]] for i in range(len(x_t)) if x_t[i] >= Vth])

        for i in range(len(point_lib)):
            seq_lib = point_lib[i]['sequence']
            x_s, y_s = seq_lib
            seq_lib_part = np.array([[x_s[i], y_s[i]] for i in range(len(x_s)) if x_s[i] >= Vth])
            dis_array[i] = e_func(seq_test_part, seq_lib_part)
    else:
        test_edge_ind = np.argmax(x_t)
        seq_test_part1 = np.array([[x_t[i], y_t[i]] for i in range(test_edge_ind + 1) if x_t[i] >= Vth])
        seq_test_part2 = np.array([[x_t[i], y_t[i]] for i in range(test_edge_ind + 1, len(x_t)) if x_t[i] >= Vth])
        if is_normalized:
            seq_test_part1 = normalized(seq_test_part1)
            seq_test_part2 = normalized(seq_test_part2)
        for i in range(len(point_lib)):
            seq_lib = point_lib[i]['sequence']
            x_s, y_s = seq_lib
            lib_edge_ind = np.argmax(x_s)
            seq_lib_part1 = np.array([[x_s[i], y_s[i]] for i in range(lib_edge_ind + 1) if x_s[i] >= Vth])
            seq_lib_part2 = np.array([[x_s[i], y_s[i]] for i in range(lib_edge_ind + 1, len(x_s)) if x_s[i] >= Vth])
            dis_array[i] = e_func(seq_test_part1, seq_lib_part1) + e_func(seq_test_part2, seq_lib_part2)            

    i_sorted = np.argsort(dis_array)
    print(i_sorted)
    outp_text += '\nSuggested params: ' + str(point_lib[i_sorted[0]]['label'])

    #-------------plot----------------#
    for i in i_sorted:
        f, ax = plt.subplots(1,1, figsize=(6,6))
        f.subplots_adjust(left=0.2, right=0.95)
        ax.scatter(seq_test[0], seq_test[1], label='Exp. data', color='red')
        ax.plot(point_lib[i]['sequence'][0], point_lib[i]['sequence'][1], label='Match')

        if transfer_mode:
            ax.set_xlabel('Gate Voltage (V)')
        else:
            ax.set_xlabel('Drain Voltage (V)')
        if is_normalized:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '|Vmax|'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', 'max'])
        if is_log:
            ax.set_yscale('log')
        if is_sqrt_mode:
            ax.set_ylabel('Sqrt Drain Current (A^0.5)')
        else:
            ax.set_ylabel('Drain Current (A)')

        img = fig2img(f)
        imgs.append((img, point_lib[i]['label']))
        plt.close('all')

    return outp_text, imgs


class interface_fit():
    def __init__(self):
        pass
    
    def title(self):
        return "Non-ideal model Atlas Matching"

    def show(self):
        return True

    def describe(self):
        description = """
        ## Quick Start
        1. Click the example below, and click `Fit` button.
        2. Drop experimental data for matching(.csv or .xlsx).
        ## Note: 
        1. A plot is generated for each value rather than each row of values in `Scanning Nonideal Parameters`.
        2. The format of a input file: first column: `Voltage`; second column: `Current`. 
        Only the first two columns would be read.
        """
        return description

    def ui(self):
        with gr.Row():
            with gr.Row():
                device_width = gr.Number(value=1000, label="W(μm)")
                device_length = gr.Number(value=10, label="L(μm)")
                device_C_ox = gr.Number(value=1e-8, label="C_ox(F/cm2)")
                device_thickness_sc = gr.Number(value=40, label="t_sc(nm)")
                voltage_fix = gr.Number(value=20, label="Vfix(V)")
                voltage_max = gr.Number(value=20, label="Vmax(V)")
                mu0 = gr.Number(value=10, label="μ1(cm2/Vs)")
                Vth0 = gr.Number(value=0.1, label="Vth0(V)")
            with gr.Row():
                scan_type = gr.Radio(['Transfer', 'Output'], value="Transfer", label="scan_type")
                is_back_mode = gr.Checkbox(value=True, label="scan back")
                is_log_mode = gr.Checkbox(value=False, label="log scale")
                is_sqrt_mode = gr.Checkbox(value=False, label="Sqrt-Current")
                is_normalized = gr.Checkbox(value=False, label="Normalized")
                algorithm = gr.Radio(['AED', 'MSE', 'LCSS', 'DTW'], value="DTW", label="Algorithm")
                semi_type = gr.Radio(['p-type', 'n-type'], value="p-type", label="semi type")
        with gr.Row():
            df = gr.Dataframe(
                label="Scanning Nonideal Parameters",
                headers=["δ-1", "barrier", "P0", "ion-mul"],
                datatype=["number", "number", "number", "number"],
                max_cols=4,
                value=[
                    [0, 0.02, 0.001, 0.02],
                    [0.3, 0.6, 0.06, 0.5]
                ],
            )
            input_file_object = gr.File(label="Input File")

        return [
            device_width, device_length, 
            device_C_ox, device_thickness_sc, voltage_fix, 
            voltage_max, mu0, Vth0, 
            scan_type, is_back_mode, is_log_mode, is_sqrt_mode, is_normalized, algorithm,
            semi_type, df, input_file_object
        ]