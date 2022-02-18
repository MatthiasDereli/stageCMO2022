

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import h5py
import os
import neurokit2 as nk
import pandas as pd

from n0_config import *

debug = False


################################
######## GET FILES ######## 
################################



def set_file(odeur, rat):
    
    if PC_ID == 'pc-jules' :
        folder_path = root + f'cmo/Etudiants/NBuonviso202201_trigeminal_sna_rat_Mathias/Data'

    if PC_ID == 'DESKTOP-G8P77K5' :
        folder_path = root 

    if odeur == 'TriG':
        file = os.path.join(folder_path, odeur, f'{rat}', f'Très TriGm-Sujets-{rat}_merge_dec_detect.hdf5')
    elif odeur == 'NonTriG':
        file = os.path.join(folder_path, odeur, f'{rat}', f'non TriGm-Sujets-{rat}_merge_dec_detect.hdf5')
        
    return file




def sig_from_hdf5(odeur, rat, sig_type):
    
    file = set_file(odeur, rat)
    
    f = h5py.File(file)
    functions = f['functions']
    if sig_type == 'ecg':
        datatype = functions['cardiac']
    elif sig_type == 'respi':
        datatype = functions['respiratory']
    elif sig_type == 'activity':
        datatype = functions['activity']
    window = datatype['window']
    signal = window['signal']

    # print(f.keys())
    # print(functions.keys())
    # print(datatype.keys())
    # print(window.keys())
    # print(signal.keys())

    if sig_type == 'ecg':
        sig = signal['ECG']['data'][:]
        # time_hdf5 = signal['ECG']['timebase'][:]
    elif sig_type == 'respi':
        sig = signal['Airflow']['data'][:]
        # time_hdf5 = signal['ECG']['timebase'][:]
    elif sig_type == 'activity':
        sig = signal['ODBA']['data'][:]
        # time_hdf5 = signal['ECG']['timebase'][:]


    time = np.arange(0 , sig.size / srates[sig_type] , 1 / srates[sig_type])

    return time, sig




################################
######## CHUNK ########
################################

#odeur, rat, sig_type, cond = 'TriG','B1','ecg', 'stress'
def chunk_sig(odeur, rat, sig_type, cond):

    if cond == 'rest':

        time, sig = sig_from_hdf5(odeur, rat, sig_type)
        if rat in rat_problem['rats']:
            chunk_stop = int(15*60*srates[sig_type])
        else:
            chunk_stop = int(21*60*srates[sig_type])
        chunk_start = int(60*srates[sig_type])
        sig_chunk = sig[chunk_start:chunk_stop]
        time_chunk = time[chunk_start:chunk_stop] - time[chunk_start:chunk_stop][0] 
        return time_chunk, sig_chunk

    elif cond == 'stress':

        time, sig = sig_from_hdf5(odeur, rat, sig_type)
        if rat in rat_problem['rats']:
            chunk_stop = int(15*60*srates[sig_type])
        else:
            chunk_stop = int(21*60*srates[sig_type])
        sig_chunk = sig[chunk_stop:]
        time_chunk = time[chunk_stop:] - time[chunk_stop:][0] 
        return time_chunk, sig_chunk
















