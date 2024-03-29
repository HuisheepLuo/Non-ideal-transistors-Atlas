import matplotlib
import matplotlib.pyplot as plt
from core.TFTmodel import TFT_property, scan_core
from core.process import extract_param, model_generation, fig2img, load_dataframe
import gradio as gr

matplotlib.use('Agg')
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Arial']

def param_input(device_width, device_length, device_C_ox, device_thickness_sc, voltage_fix, 
                voltage_max, mu0, Vth0, scan_type, wav_type, is_log, is_sqrt_mode, is_normalized, df_in):
    TFT_prop = TFT_property(device_width, device_length, device_C_ox, device_thickness_sc)

    voltage_step = voltage_max / 100

    if scan_type.lower() == 'transfer':
        transfer_mode = True
    else: transfer_mode = False

    s_core = scan_core(transfer_mode, voltage_fix, voltage_max, voltage_step,
                    property=TFT_prop, wav_type=wav_type)
    
    TFT_prop.renew_param('mu10', mu0)
    TFT_prop.renew_param('mu20', mu0)
    s_core.renew_param('Vth', Vth0)

    if transfer_mode:
        x_name = 'VG'
    else:
        x_name = 'VD'

    if wav_type  == '1xspk' or wav_type  == '5xspk':
        x_name = 't_array'

    if is_sqrt_mode:
        y_name = 'I_tot_sqrt'
    else:
        y_name = 'I_tot'

    param_group = load_dataframe(df_in)
    point_lib = model_generation(param_group, TFT_prop, s_core, x_name, y_name, is_normalized)

    plt.close('all')
    counts = len(point_lib)
    imgs = []
    for i in range(counts):
        f, ax = plt.subplots(1,1, figsize=(6,6))
        f.subplots_adjust(left=0.2, right=0.95)
        if wav_type == 'tri':
            ax.plot(point_lib[i]['sequence'][0][:100], point_lib[i]['sequence'][1][:100])
            ax.plot(point_lib[i]['sequence'][0][100:], point_lib[i]['sequence'][1][100:])
        else:
            ax.plot(point_lib[i]['sequence'][0], point_lib[i]['sequence'][1])

        if transfer_mode:
            ax.set_xlabel('Gate Voltage (V)')
        else:
            ax.set_xlabel('Drain Voltage (V)')

        if wav_type == '1xspk' or wav_type  == '5xspk':
            ax.set_xlabel('Time (s)')

        if is_normalized:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '|max|'])
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
        
    return imgs


class interface_lib():
    def __init__(self):
        pass
    
    def title(self):
        return "Non-ideal model Atlas"

    def show(self):
        return True

    def describe(self):
        description = """
        ## Quick Start
        Enter parameters for generating Atlas.
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
                wav_type = gr.Radio(['tri', 'lin', '1xspk', '5xspk'], value="tri", label="wave_type")
                # is_back_mode = gr.Checkbox(value=True, label="scan back(Triangular wave)")
                is_log_mode = gr.Checkbox(value=False, label="log scale")
                is_sqrt_mode = gr.Checkbox(value=False, label="Sqrt-Current")
                is_normalized = gr.Checkbox(value=False, label="Normalized")
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
        return [
            device_width, device_length, 
            device_C_ox, device_thickness_sc, voltage_fix, 
            voltage_max, mu0, Vth0, 
            scan_type, wav_type, is_log_mode, is_sqrt_mode, is_normalized, df
        ]
