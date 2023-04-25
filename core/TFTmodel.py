import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math
import pandas as pd
from core.utils import data_normalization, localsToDict
import os

'''
developed by Yiyang Luo and Prof. Liu
version 0.14


'''
class TFT_property:
    '''
    `TFT_property` is a class for storing the property or parameters of the device.
    '''

    def __init__(self, device_width, device_length, device_C_ox, device_thickness_sc):
        '''
        device_width: Channel width-W(um), 
        device_length: Channel length-L(um), 
        device_C_ox: Oxide layer Capacitance-Cox(F/cm2), 
        device_thickness_sc: Thickness of semiconductor layer-tsc(nm)
        '''
        self.width, self.length, self.Cox, self.tsc = device_width, device_length, device_C_ox, device_thickness_sc

        # mobility
        self.mu10 = 10 #cm2/Vs
        self.mu1_power = 0
        self.mu20 = 10 #cm2/Vs
        self.mu2_power = 0
        # contact
        self.Q000 = 1.3e-12 #C/cm3
        self.Q0_power = 1
        self.barrier = 0.1 #eV
        self.d = 1 #cm
        self.alpha = 2
        self.rho = 1.0e12 # Ohm
        # trapping
        self.P0 = 0.001 #s^-1
        self.kappa = 0
        #----------------- ion ------------------#
        self.ion_multiple = 0.02
        self.Di0 = 6.67e-16
        self.Di_Di0_ratio = 0.02
        # positive ion
        self.pos_beta0 = 1
        self.pos_beta_mul = 0
        self.pos_tau1 = 1.02e-1
        self.pos_tau0 = 1.9e-14
        self.pos_tau_alpha = 1
        self.pos_tau_gamma = 1
        # negative ion
        self.neg_beta0 = 1
        self.neg_beta_mul = 0
        self.neg_tau1 = 1.0
        self.neg_tau0 = 1.35e-15
        self.neg_tau_alpha = 1
        self.neg_tau_gamma = 1
        #
        self.epsilon_r = 9
        self.epsilon0 = 8.85e-14
        # fitting
        self.a = 0.25 #eV
        self.Q00_m = 1.3e-10 #C/cm3
        self.m = 1
        self.Qtrap_max = 3e-4
        self.T = 300 #K
        self.secondary_param()

        # secondary
    def secondary_param(self):
        '''
        Calculate secondary parameter
        '''
        # 
        self.kT_q = 1.38e-23 * self.T / 1.6e-19 #eV
        self.S = self.width * 1e-4 * self.tsc * 1e-7 #cm2
        self.WC_L = self.width / self.length * self.Cox
        self.beta = self.alpha * 2 - 1
        self.Rback = self.rho / self.tsc * self.length / self.width
        self.delta = 1 + self.mu1_power
        # ion
        self.conc_n0 = 2e20 * self.ion_multiple
        self.pos_conc_total0 = 1.4e21 * self.ion_multiple
        self.pos_A1 = -6e20 * self.ion_multiple
        self.neg_conc_total0 = 2e17 * self.ion_multiple
        self.neg_A1 = 0
        self.Di = self.Di0 * self.Di_Di0_ratio
        self.ion_gamma = 2 * self.conc_n0 / self.pos_conc_total0
        self.Ld = 1 / math.sqrt(4 * np.pi / self.kT_q * 1.6e-19 / self.epsilon0 / self.epsilon_r * 2 * self.conc_n0)
        self.C_ion_0 = self.epsilon0 * self.epsilon_r / (4 * np.pi) / self.Ld

    def renew_param(self, paramName, value):
        '''
        Change a parameter's value in this class during the code runing. 
        '''
        try:
            exec('self.' + paramName + '=' + str(value))
            self.secondary_param()
        except:
            raise ValueError
        return

class scan_core:
    '''
    generating point
    '''
    def __init__(self,transfer_mode:bool, voltage_fix, voltage_max, voltage_step, property:TFT_property,back:bool=True, scan_time_step=1):
        '''
        transfer_mode: Scanning Gate voltage(True) or Drain voltage(False).
        voltage_fix(V): Fixed voltage during scanning. If transfer_mode is True(VG), then VD will be fixed at this value.
        voltage_max(V): Maximum scanning voltage.
        voltage_step(V/s): Scanning step.
        scan_time_step(s): Scanning time of one step. Default: 1s
        property: a 'TFT_property' class input.
        back: Whether scanning cycle for hysteresis or not.

        '''
        self.V_fix = voltage_fix
        self.V_max = voltage_max
        self.V_step = voltage_step # V_step(V/s)
        self.t_step = scan_time_step
        self.trans = transfer_mode
        self.Vg_minus_Vth_min = 1e-4
        self.property = property
        self.Vth = 0
        self.back = back
        self.I_0 = 9e-12
        self.Vth0 = self.Vth
        self.Von = self.Vth + 0.015
        self.VT0 = self.Vth + 0.05
        self.PZC = 0 # zero charge potential
        self.generate_point()

    def renew_property(self, property):
        '''
        Change property in this class during the code runing. 
        '''
        self.property = property
        self.generate_point()
        return

    def renew_param(self, paramName, value):
        '''
        Change a parameter's value in this class during the code runing. 
        Can not change property.
        '''
        exec('self.' + paramName + '=' + str(value))
        self.generate_point()
        return

    def generate_point(self):
        '''
        Calculation of all points.
        '''
        if self.back:
            self.V_scan_array = np.concatenate((np.arange(0, self.V_max, self.V_step*self.t_step), 
                    np.arange(self.V_max, 0, -self.V_step*self.t_step)),axis=0) + self.Vg_minus_Vth_min

        else:
            self.V_scan_array = np.arange(0, self.V_max, self.V_step*self.t_step) + self.Vg_minus_Vth_min
        self.step = len(self.V_scan_array)
        step = self.step
        self.t_max = step * self.t_step
        self.t_array = np.arange(0, self.t_max, self.t_step)
        self.t_array[0] = self.t_step / 2
        self.V_fix_array = np.full((step),self.V_fix)
        # init array
        prop = self.property
        VD_real = np.zeros(step)
        P = np.zeros(step)
        Q = np.zeros(step)
        Vth = np.zeros(step)
        VG_minus_Vth_real = np.zeros(step)
        mu1, mu2 = np.zeros(step), np.zeros(step) 
        Q0_1, Q0_2 = np.zeros(step), np.zeros(step)
        gamma_p = np.zeros(step)
        A, B, a, b, c = np.zeros(step), np.zeros(step), np.zeros(step), np.zeros(step), np.zeros(step)
        bb_4ac = np.zeros(step)
        Vs = np.zeros(step)
        S, D = np.zeros(step), np.zeros(step)
        I_RC = np.zeros(step)
        I_tft = np.zeros(step)
        R_tot = np.zeros(step)
        I_sub = np.zeros(step)
        I_tot = np.zeros(step)
        I_tot_sqrt = np.zeros(step)
        Vx1 = np.zeros(step)
        Vx2 = np.zeros(step)
        Rch = np.zeros(step)
        Rs = np.zeros(step)
        mu_lin = np.zeros(step)
        mu_sat = np.zeros(step)
        delta = prop.delta

        # ion
        conc_total_n, tau_n, beta_n, conc_n = np.zeros(step), np.zeros(step), np.zeros(step), np.zeros(step)
        dndt, conc_n_from_dndt = np.zeros(step), np.zeros(step)
        conc_total_p, tau_p, beta_p = np.zeros(step), np.zeros(step), np.zeros(step)
        dpdt, conc_p_from_dpdt = np.zeros(step), np.zeros(step)
        n_n0_ratio, u0, sinh2u, C_ion, Q_ion = np.zeros(step), np.zeros(step), np.zeros(step), np.zeros(step), np.zeros(step)
        Von, VT0 = np.zeros(step), np.zeros(step)
        SS_formula = np.zeros(step)

        if self.trans: # transfer
            VG = self.V_scan_array
            VD = self.V_fix_array
        else: # output
            VG = self.V_fix_array
            VD = self.V_scan_array
        for i in range(0,step):
            # ion capacitance
            if VG[i] >= self.PZC:
                conc_total_n[i] = prop.pos_conc_total0 + prop.pos_A1 \
                            * math.exp(-abs(VG[i]) / prop.pos_tau1)
                tau_n[i] = prop.pos_tau0 / math.pow(abs(VG[i]), prop.pos_tau_alpha) \
                            / math.pow(prop.Di, prop.pos_tau_alpha)
                beta_n[i] = prop.pos_beta0 * math.pow(abs(VG[i]), prop.pos_beta_mul)
                conc_total_p[i] = prop.neg_conc_total0 + prop.neg_A1 \
                            * math.exp(-abs(VG[i]) / prop.neg_tau1)
                tau_p[i] = prop.neg_tau0 / math.pow(abs(VG[i]), prop.neg_tau_alpha) \
                            / math.pow(prop.Di, prop.neg_tau_alpha)
                beta_p[i] = prop.neg_beta0 * math.pow(abs(VG[i]), prop.neg_beta_mul)        
            else:
                conc_total_n[i] = prop.neg_conc_total0 + prop.neg_A1 \
                            * math.exp(-abs(VG[i]) / prop.neg_tau1)
                tau_n[i] = prop.neg_tau0 / math.pow(abs(VG[i]), prop.neg_tau_alpha) \
                            / math.pow(prop.Di, prop.neg_tau_alpha)
                beta_n[i] = prop.neg_beta0 * math.pow(abs(VG[i]), prop.neg_beta_mul)
                conc_total_p[i] = prop.pos_conc_total0 + prop.pos_A1 \
                            * math.exp(-abs(VG[i]) / prop.pos_tau1)
                tau_p[i] = prop.pos_tau0 / math.pow(abs(VG[i]), prop.pos_tau_alpha) \
                            / math.pow(prop.Di, prop.pos_tau_alpha)
                beta_p[i] = prop.pos_beta0 * math.pow(abs(VG[i]), prop.pos_beta_mul)                          
            conc_n[i] = conc_total_n[i] + (prop.conc_n0 - conc_total_n[i]) \
                        / math.exp(math.pow(self.t_array[i] / tau_n[i], beta_n[i])) 
            if i == 0:
                conc_n_from_dndt[i] = prop.conc_n0
                conc_p_from_dpdt[i] = prop.conc_n0
            else:
                conc_n_from_dndt[i] = conc_n_from_dndt[i-1] + dndt[i-1] \
                                    * (self.t_array[i] - self.t_array[i-1])
                conc_p_from_dpdt[i] = conc_p_from_dpdt[i-1] + dpdt[i-1] \
                                    * (self.t_array[i] - self.t_array[i-1])
            dndt[i] = (conc_total_n[i] - conc_n_from_dndt[i]) * math.pow(1 / tau_n[i], beta_n[i]) \
                    * beta_n[i] * math.pow(self.t_array[i], beta_n[i] - 1)
            dpdt[i] = (conc_total_p[i] - conc_p_from_dpdt[i]) * math.pow(1 / tau_p[i], beta_p[i]) \
                    * beta_p[i] * math.pow(self.t_array[i], beta_p[i] - 1)  
            n_n0_ratio[i] = conc_n_from_dndt[i] / prop.conc_n0

            u0[i] = math.log((-n_n0_ratio[i] + n_n0_ratio[i] * prop.ion_gamma \
                    - math.sqrt(math.pow(n_n0_ratio[i], 2) + 2 * prop.ion_gamma \
                    * n_n0_ratio[i] - 2 * math.pow(n_n0_ratio[i], 2) * prop.ion_gamma)) \
                    / (-2 + n_n0_ratio[i] * prop.ion_gamma))
            sinh2u[i] = 1 + 2 * prop.ion_gamma * math.pow(math.sinh(u0[i] / 2), 2)
            if u0[i] != 0 and u0[i] >= 0.0001:
                C_ion[i] = math.cosh(u0[i] / 2) / sinh2u[i] \
                        * math.pow((sinh2u[i] - 1) / math.log(sinh2u[i]), 0.5) * prop.C_ion_0
            else:
                C_ion[i] = 1 * prop.C_ion_0
            Q_ion[i] = 1.6e-19 * 1e-7 * (conc_n_from_dndt[i] - conc_p_from_dpdt[i])     
            
        for i in range(0, step):
            # trapping
            if i == 0 :
                P[i] = 0
                Q[i] = P[i]
            else:
                if prop.P0 > 0:
                    P[i] = prop.P0 * I_tot[i-1] * (self.t_array[i] - self.t_array[i-1]) * prop.width / prop.length * math.pow(self.t_array[i], prop.kappa)
                else:
                    P[i] = prop.P0 * math.pow(self.t_array[i], prop.kappa)
                Q[i] = min(Q[i-1] + (prop.Qtrap_max - Q[i-1]) * (self.t_array[i] - self.t_array[i-1])  * P[i], abs(prop.Qtrap_max))
            Vth[i] = (Q[i] * prop.P0 - Q_ion[i]) / C_ion[i] + self.Vth
            Von[i] = Vth[i] + self.Von
            VT0[i] = Vth[i] + self.VT0
            VG_minus_Vth_real[i] = max(VG[i]-Vth[i], self.Vg_minus_Vth_min)
            if VD[i] < VG_minus_Vth_real[i]:
                VD_real[i] = VD[i]
            else: VD_real[i] = VG_minus_Vth_real[i]

            # mobility
            mu1[i] = prop.mu10 * math.pow(VG_minus_Vth_real[i], prop.mu1_power)
            mu2[i] = prop.mu20 * math.pow(VG_minus_Vth_real[i], prop.mu2_power)
            try:
                Q0_1[i] = prop.Q000 * math.exp((-prop.barrier) / prop.kT_q) * math.exp(math.pow(VG_minus_Vth_real[i] / prop.a, prop.Q0_power))
                Q0_2[i] = math.pow(math.pow(Q0_1[i], -prop.m) + math.pow(prop.Q00_m, -prop.m), -prop.m)
            except:
                raise ValueError
            gamma_p[i] = Q0_2[i] * prop.S * prop.mu20 / math.pow(prop.d * 1e-4, prop.beta) / (prop.width * C_ion[i] / prop.length) / prop.mu10
            A[i] = VG_minus_Vth_real[i] * 2 - VD_real[i]
            B[i] = math.pow(2, delta) * gamma_p[i]
            a[i] = B[i] - math.pow(A[i], delta - 1) * delta
            b[i] = math.pow(A[i], delta) + math.pow(A[i], delta - 1) * delta * VD_real[i]
            c[i] = - math.pow(A[i], delta) * VD_real[i]
            bb_4ac[i] = b[i] * b[i] - 4 * a[i] * c[i]
            if bb_4ac[i] > 0:
                Vs[i] = (-b[i] + math.sqrt(bb_4ac[i])) / 2 / a[i]
            else:
                Vs[i] = 0
            S[i] = (VG_minus_Vth_real[i] - Vs[i]) / VG_minus_Vth_real[i]
            if (VG_minus_Vth_real[i] - VD_real[i]) / (VG_minus_Vth_real[i] - Vs[i]) > 0:
                D[i] = (VG_minus_Vth_real[i] - VD_real[i]) / (VG_minus_Vth_real[i] - Vs[i])
            else:
                D[i] = 0
            I_RC[i] = Q0_2[i] * mu2[i] * math.pow(Vs[i], prop.alpha) * prop.S / math.pow(prop.d * 1e-4, prop.beta)
            I_tft[i] = max((prop.width * C_ion[i] / prop.length) * prop.mu10 * math.pow(VG_minus_Vth_real[i] - (VD_real[i] + Vs[i]) / 2, delta) * (VD_real[i] - Vs[i]), 0)
            if I_tft[i] == 0:
                R_tot[i] = prop.Rback / VG_minus_Vth_real[i]
            else:
                R_tot[i] = prop.Rback / VG_minus_Vth_real[i] * VD[i] / I_tft[i] / (prop.Rback / VG_minus_Vth_real[i] + VD[i] / I_tft[i])
            try:
                I_sub[i] = self.I_0 * prop.width / prop.length \
                            * math.log(1 + math.exp(math.log(10) * VG_minus_Vth_real[i] / 0.08)) \
                            * (1 - math.exp(-math.log(10) * VD[i] / 0.08)) \
                            + prop.width / prop.length * 1e-13 * VD_real[i]
            except:
                I_sub[i] = 0
            I_tot[i] = VD[i] / R_tot[i] * (1 + math.tanh(2 * (VG[i] - VT0[i]))) / 2 \
                        + I_sub[i] * (1 - math.tanh(2 * (VG[i] - VT0[i]))) / 2

            I_tot_sqrt[i] = math.sqrt(abs(I_tot[i]))
            Rch[i] = (VD[i] - Vs[i]) / I_tot[i]
            Rs[i] = Vs[i] / I_tot[i]


            if (i != 0) & (i != 1) & self.trans:
                if (VG_minus_Vth_real[i] - VG_minus_Vth_real[i-1]) != 0:
                    mu_lin[i] = (I_tot[i] - I_tot[i-1]) / (VG_minus_Vth_real[i] - VG_minus_Vth_real[i-1]) / VD[i] / prop.WC_L
                if (VG[i] - VG[i-1]) != 0 and I_tot[i] > 0 and I_tot[i-1] > 0:   
                    mu_sat[i] = math.pow((math.sqrt(I_tot[i]) - math.sqrt(I_tot[i-1])) / (VG[i] - VG[i-1]), 2) * 2 / prop.WC_L
            
            Vx1[i] = VG_minus_Vth_real[i] * (1 - S[i] * (math.pow(1 - (1 - math.pow(D[i], delta + 1)) / 3, 1 / (delta + 1))))
            Vx2[i] = VG_minus_Vth_real[i] * (1 - S[i] * (math.pow(1 - (1 - math.pow(D[i], delta + 1)) * 2 / 3, 1 / (delta + 1))))

        def Vx_Vg(x):
            '''
            Function for measuring channel potential.
            Return 1D ndarray.
            '''
            Vx = np.zeros(step)
            for i in range(0,step):
                Vx[i] = VG_minus_Vth_real[i] * (1 - S[i] * (math.pow(1 - (1 - math.pow(D[i], delta + 1)) * x / prop.length, 1 / (delta + 1))))

            return Vx
        # point tuples
        self.I_Vg_curve_point = [self.V_scan_array, I_tot]
        self.I_Vg_curve_point_log = [self.V_scan_array, np.log10(abs(I_tot))]
        self.Vx1_point = [self.V_scan_array, Vx_Vg(prop.length / 3)]
        self.Vx2_point = [self.V_scan_array, Vx_Vg(prop.length / 3 * 2)]
        # Y normalization of point tuples
        self.norm_I_Vg_curve_point = [self.V_scan_array, data_normalization(I_tot)]
        if self.trans:
            self.mu_Vg_curve_point = [self.V_scan_array, mu_sat] 
            self.norm_mu_Vg_curve_point = [self.V_scan_array, data_normalization(mu_sat)]
        self.norm_Vx1_point = [self.V_scan_array, (Vx_Vg(prop.length / 3) - np.min(Vx_Vg(prop.length / 3 * 2)))
                                / (np.max(Vx_Vg(prop.length / 3 * 2)) - np.min(Vx_Vg(prop.length / 3 * 2)))]
        self.norm_Vx2_point = [self.V_scan_array, data_normalization(Vx_Vg(prop.length / 3 * 2))]
        self.norm_Vmid_point = [self.V_scan_array, data_normalization(Vx_Vg(prop.length / 2))]
        self.norm_Id_Vmid_point = [data_normalization(I_tot), data_normalization(Vx_Vg(prop.length / 2))]
            
        # dict of all parameters in locals(only class "ndarray")
        self.dict = localsToDict(locals().copy())
        
        return

    def csvout(self, out_file_name:str='out.csv'):
        '''
        Generate csv file based on points' data.
        '''
        df = pd.DataFrame(self.dict)
        df.to_csv(out_file_name)







