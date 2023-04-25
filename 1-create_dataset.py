import pandas as pd
import numpy as np
import copy

import sys
import os
sys.path.append(os.getcwd())
from core.TFTmodel import *
from core.utils import timefn

#-----------------TFT parameters preset-------------------#
TFT_sample = TFT_property(device_width=1000, device_length=30, device_C_ox=2e-7, device_thickness_sc=40)
scan_core_sample = scan_core(transfer_mode=True, voltage_fix=20, voltage_max=20, voltage_step=0.2,
                scan_time_step=1, property=TFT_sample, back=True)

#-----------------scanning parameters-----------------#
# stru params: device_width & device_length
stru_params = [[1000,300], [1000,100], [1000,30], [1000,10], [1000,3]]

# scan params: Vfix & Vmax
scan_params = [[0.1, 0.1], [0.1, 10], [0.1, 20], [0.1, 50], [10, 10], [10, 20], [10, 50], [20, 20], [20, 50], [50, 50]]

# non-ideal params
TFT_sample.renew_param('mu1_power', 0.0)
TFT_sample.renew_param('P0', 0.0001)
TFT_sample.renew_param('barrier', 0.0)
TFT_sample.renew_param('ion_multiple', 0.02)

train_param_group = {
    'mu1_power': np.linspace(0.1, 0.5, 50), 
    'P0': np.logspace(-2, 0, 50), 
    'barrier': np.linspace(0.1, 1, 50),
    'ion_multiple': np.linspace(0.02, 20, 50), 
}



@timefn
def create_chart_dataset(out_label_name:str, TFTmodel:TFT_property, scan_core:scan_core, param_group:dict, mode:str):
    """
    Generate chart data

    Args:
        out_label_name(str): output dirpath 
        TFTmodel: class `TFTmodel.TFT_property`
        scan_core: class `scan_core`
        param_group: dict of parameter group of TFT_property, values should be list() or np.ndarray().
        mode(str): Data to be generated, such as `i`, `i_norm`, `mu`, `v1`, `v2`. Case-insensitive. 

    Returns:
        list of class chart_lib: [chart_lib1, chart_lib2, ...].
    """

    lib = []
    count = 0
    data_list = []
    delta_list = []
    P0_list = []
    barrier_list = []
    ratio_list = []
    type_value = []

    for W, L in stru_params:
        for Vfix, Vmax in scan_params:
            TFT_1 = copy.deepcopy(TFTmodel)
            TFT_1.renew_param('device_width', W)
            TFT_1.renew_param('device_length', L)
            scan_core.renew_param('voltage_fix', Vfix)
            scan_core.renew_param('voltage_max', Vmax)

            for param, values in param_group.items():
                param_values = list()
                for i in range(np.size(values)):
                    if param == 'mu1_power':
                        TFT_1.renew_param(param, values[i])
                        param_values = [values[i], 0.0001, 0.1, 0.02, 0]
                    elif param == 'P0':
                        TFT_1.renew_param(param, values[i])
                        param_values = [0, values[i], 0.1, 0.02, 1]
                    elif param == 'barrier':
                        TFT_1.renew_param(param, values[i])
                        param_values = [0, 0.0001, values[i], 0.02, 2]
                    elif param == 'ion_multiple':
                        TFT_1.renew_param(param, values[i])
                        param_values = [0, 0.0001, 0.1, values[i], 3]
                    scan_core.renew_property(TFT_1) # this may cause problem.

                    if mode.lower() == 'mu': # mu-V with data normalization
                        c=chart_lib(scan_core.norm_mu_Vg_curve_point, param_values)
                    elif mode.lower() == 'i': # I-V
                        c=chart_lib(scan_core.I_Vg_curve_point, param_values)
                    elif mode.lower() == 'i_norm': # I-V
                        c=chart_lib(scan_core.norm_I_Vg_curve_point, param_values)
                    elif mode.lower() == 'i_log': # logI-V
                        c=chart_lib(scan_core.I_Vg_curve_point_log, param_values)
                    elif mode.lower() == 'v1': # V1-V
                        c=chart_lib(scan_core.Vx1_point, param_values)
                    elif mode.lower() == 'v2': # V2-V
                        c=chart_lib(scan_core.Vx2_point, param_values)
                    else:
                        return print("Wrong mode type.")

                    data_list.append(c.point[1])
                    delta_value, P0_value, barrier_value, ratio_value, main_factor = param_values
                    delta_list.append(delta_value)
                    P0_list.append(P0_value)
                    barrier_list.append(barrier_value)
                    ratio_list.append(ratio_value)
                    type_value.append(main_factor)
                    lib.append(c)
                    count += 1

    df = pd.DataFrame(np.array(data_list))
    df2 = pd.DataFrame({
                    'delta':delta_list, 
                    'P0':P0_list, 
                    'phi':barrier_list,
                    'ion':ratio_list,
                    'type': type_value,
                    })
    df = df.join(df2)
    df.to_csv(out_label_name+'.csv', index=False)
    print(f'Successfully generated {count} csv files.')
    return lib


class chart_lib:
    '''
    Class for storing charts' input and output.
    '''
    count = 0
    def __init__(self, point:list or np.ndarray, param_values:list):
        self.point = point
        # self.delta, self.P0, self.barrier, self.Di_Di0_ratio, self.main_factor = param_values
        self.labels = param_values
        chart_lib.count += 1
     
    def csv_out(self, out_dir:str):
        df = pd.DataFrame({'point_x':self.point[0], 'point_y':self.point[1]})
        file_name = str(chart_lib.count) + '.csv'
        df.to_csv(out_dir + '\\' + file_name, index=False)
        return file_name

if __name__ == '__main__':
    train_lib = create_chart_dataset('dataset\\train_set', TFT_sample, scan_core_sample, train_param_group, 'i_norm')




