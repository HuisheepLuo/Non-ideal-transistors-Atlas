import numpy as np
import pandas as pd
import copy
from itertools import product
from scipy import optimize, signal, diff
from PIL import Image
import math
from numba import njit, jit, float64
from core.utils import data_normalization

def model_generation(param_group:dict, TFT_model, scan_core, x:str, y:str, is_normalized:bool, is_comb=False):
    # storing dicts of models and parameters
    point_lib = []
    # Combination of various parameters' values
    # from dict({'a':[1,2,3],'b':[1,2]}) to dict({'a':[1,1,2,2,3,3],'b':[1,2,1,2,1,2]})
    if is_comb:
        param_group_values = {
            k: list(v) for k, v in zip(
                param_group.keys(), 
                np.transpose(
                    list(
                        product(*param_group.values())
                    )
                )
            )
        }

        # After combination all key values in param_group_values have the same length.
        for i in range(len(param_group_values['P0'])):
            param_values = list()
            _TFT_model = copy.deepcopy(TFT_model)
            for param, values in param_group_values.items():
                _TFT_model.renew_param(param, values[i])
                param_values.append(values[i])

            _scan_core = copy.deepcopy(scan_core)
            _scan_core.renew_property(_TFT_model) 
            label_name = 'δ: ' + str(param_values[0]) + ', φ: ' + str(param_values[2])\
                        + ', P0: ' + str(param_values[1]) +', pi0: '+str(_TFT_model.conc_n0)
            if is_normalized:
                seq = np.array([data_normalization(_scan_core.dict[x]), data_normalization(_scan_core.dict[y])])
            else:
                seq = np.array([_scan_core.dict[x], _scan_core.dict[y]])
            point_lib.append(dict({
                'sequence': seq, 
                'param': param_values,
                'core': _scan_core,
                'label': label_name
            }))

        return point_lib
    else:
        for param, values in param_group.items():
            param_values = list()
            _TFT_model = copy.deepcopy(TFT_model)
            for i in range(np.size(values)):
                if param == 'mu1_power':
                    _TFT_model.renew_param(param, values[i])
                    param_values = [values[i], 0.001, 0.02, 0.005, 0]
                elif param == 'P0':
                    _TFT_model.renew_param(param, values[i])
                    param_values = [0, values[i], 0.02, 0.005, 1]
                elif param == 'barrier':
                    _TFT_model.renew_param(param, values[i])
                    param_values = [0, 0.001, values[i], 0.005, 2]
                elif param == 'ion_multiple':
                    _TFT_model.renew_param(param, values[i])
                    param_values = [0, 0.001, 0.02, values[i], 3]

                _scan_core = copy.deepcopy(scan_core)
                _scan_core.renew_property(_TFT_model) 
                label_name = 'δ: ' + str(param_values[0]+1) + ', φ: ' + str(param_values[2])\
                            + ', P0: ' + str(param_values[1]) +', pi0: '+str(_TFT_model.conc_n0)
                if is_normalized:
                    seq = np.array([data_normalization(_scan_core.dict[x]), data_normalization(_scan_core.dict[y])])
                else:
                    seq = np.array([_scan_core.dict[x], _scan_core.dict[y]])
                point_lib.append(dict({
                    'sequence': seq, 
                    'param': param_values,
                    'core': _scan_core,
                    'label': label_name
                }))

        return point_lib

def linear_func(x, k, b):
    return k * x + b

def extract_param(sequence, TFT_model):
    x0, y0 = sequence
    y0 = np.sqrt(abs(y0))
    x_inflection = x0[signal.argrelextrema(diff(y0), np.greater)][-1]
    # print(x_inflection)
    x, y = np.array([[x0[i],y0[i]] for i in range(len(x0)) if x0[i] >= x_inflection]).T
    popt, pcov = optimize.curve_fit(linear_func, x, y)
    k, b = popt
    mobility = 2 * TFT_model.length / TFT_model.width / TFT_model.Cox * k ** 2
    Vth = b / (-k)
    # plt.figure()
    # plt.scatter(x0, y0)
    # plt.plot(x, linear_func(x, k, b))
    return mobility, Vth, x_inflection

def normalized(seq: np.ndarray):
    '''
    normalize xs and normalize ys
    seq: [[x1,y1],[x2,y2],...]
    return: [[norm_x1,norm_y1],[norm_x2,norm_y2],...]
    '''
    return np.array([data_normalization(seq.T[0]),data_normalization(seq.T[1])]).T

def load_dataframe(df: pd.DataFrame):
    def no_null(i):
        alist = df.iloc[:,i].to_list()
        alist_new = []
        for j in alist:
            if j != '':
                alist_new.append(float(j))
        return np.array(alist_new)

    param_group = {
        'mu1_power':no_null(0), 
        'barrier':no_null(1), 
        'P0':no_null(2), 
        'ion_multiple':no_null(3),
    }
    print(param_group)
    return param_group


def fig2img(fig):
    '''
    Figure to PIL-Image
    '''
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes('RGB', (w,h), fig.canvas.tostring_rgb())
    return img

@njit(float64(float64[:],float64[:]))
def euclidean_dis(pa, pb):
    '''
    Eucildean distance of point A and point B in 2d.

    pa = (xa, ya);
    pb = (xb, yb)
    '''
    xa, ya = pa
    xb, yb = pb
    return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

@jit(float64(float64[:,:],float64[:,:]))
def DTW(s1, s2):
    '''
    Dynamic Time Warping (DTW)
    '''

    m = len(s1)
    n = len(s2)

    dp = np.zeros((m, n), dtype=np.float64)

    # Edge of distance matrix
    for i in range(m):
        dp[i,0] = euclidean_dis(s1[i], s2[0])
    for j in range(n):
        dp[0,j] = euclidean_dis(s1[0], s2[j])
    dp[0,0] = 0
    for i in range(1, m):
        for j in range(1, n):
            dp[i,j] = min(dp[i-1,j-1], dp[i-1,j], dp[i,j-1]) + euclidean_dis(s1[i], s2[j])

    return dp[-1,-1]

@jit(float64(float64[:,:],float64[:,:]))
def AED(s1, s2):
    '''
    Average Euclidean Distance (AED)
    '''
    d = 0.0
    m = len(s1)
    for i in range(m):
        d = d + euclidean_dis(s1[i], s2[i])
    d = d / m
    return d

@jit(float64(float64[:,:],float64[:,:]))
def MSE(s1, s2):
    '''
    Mean Squared Error (MSE)
    '''
    d = 0.0
    m = len(s1)
    for i in range(m):
        d = d + (s1[i,1] - s2[i,1]) ** 2
    d = d / m
    return d

@jit(float64(float64[:,:],float64[:,:]))
def LCSS(s1, s2):
    '''
    Longest common sub-sequence (LCSS)
    '''
    
    m = len(s1)
    n = len(s2)

    sigma_x = 0.4
    sigma_y = 0.1
    delta = 5

    dp = np.zeros((m, n), dtype=float64)

    # edge
    for i in range(m):
        dp[i,0] = 0
    for j in range(n):
        dp[0,j] = 0
    
    for i in range(1, m):
        for j in range(1, n):
            if abs(s1[i,0]-s2[j,0]) < sigma_x and abs(s1[i,1]-s2[j,1]) < sigma_y and abs(i-j) < delta:
                dp[i,j] = dp[i-1,j-1] + 1
            else:
                dp[i,j] = max(dp[i-1,j], dp[i,j-1])

    return -dp[-1,-1]

@jit(float64(float64[:,:],float64[:,:]))
def EDR(s1, s2):
    '''
    Edit Distance on Real Sequence (EDR).
    '''
    
    m = len(s1)
    n = len(s2)

    epsilon = 0.2

    dp = np.zeros((m+1, n+1), dtype=float64)

    # edge
    for i in range(m+1):
        dp[i,0] = i
    for j in range(n+1):
        dp[0,j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            subcost = (euclidean_dis(s1[i-1], s2[j-1]) > epsilon)
            dp[i,j] = min(dp[i-1,j-1] + subcost, dp[i-1,j] + 1, dp[i,j-1] + 1)

    return dp[-1,-1]