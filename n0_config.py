

import numpy as np
import scipy.signal



########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()


if PC_ID == 'pc-jules':

    root = '/home/jules/smb4k/CRNLDATA/crnldata/'
    n_core = 6

elif PC_ID == 'DESKTOP-G8P77K5':

    root = r'C:\Users\manip\Desktop\Matthias_data'
    n_core = 1

path_save = os.path.join(root, 'cmo', 'Etudiants', 'NBuonviso202201_trigeminal_sna_rat_Mathias', 'Analyses')
path_savefig = os.path.join(path_save, 'figures') 




########################
######## PARAMS ########
########################



srates = {'ecg':500,'respi':200,'activity':12.5}


rats = ['B1','B2','N1','N2','R1','R2','V1','V2']
odeurs = ['TriG','NonTriG']
sig_types = ['ecg','respi','activity']

conditions = ['rest', 'stress']

hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2', 'HRV_LFHF']

stress_time_to_take = 10 #in sec
n_noise_stim = 15

rat_problem = {'rats' : ['R1'], 'session' : ['TriG']}




################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)






