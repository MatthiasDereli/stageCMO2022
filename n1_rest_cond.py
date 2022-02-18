

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import h5py
import os
import neurokit2 as nk
import pandas as pd
from bycycle.cyclepoints import find_extrema
import seaborn as sns

from n0_config import *
from n0bis_analysis_functions import *

debug = False






########################################
######## HRV ANALYSIS HOMEMADE ########
########################################

#### RRI, IFR


#ecg_i, ecg_cR, srate, srate_resample = ecg_i, ecg_cR, srates['ecg'], srate_resample_hrv
def get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample) :

    cR_sec = ecg_cR # cR in sec
    times = np.arange(0,len(ecg_i))/srate # in sec

    # RRI computation
    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))
    IFR = (1/RRI)


    # interpolate
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic')
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    #plt.plot(cR_sec, RRI, label='old')
    #plt.plot(cR_sec_resample, RRI_resample, label='new')
    #plt.legend()
    #plt.show()


    # figure
    fig, ax = plt.subplots()
    ax = plt.subplot(411)
    plt.plot(times, ecg_i)
    plt.title('ECG')
    plt.ylabel('a.u.')
    plt.xlabel('s')
    plt.vlines(cR_sec, ymin=min(ecg_i), ymax=max(ecg_i), colors='k')
    plt.subplot(412, sharex=ax)
    plt.plot(cR_sec, RRI)
    plt.title('RRI')
    plt.ylabel('s')
    plt.subplot(413, sharex=ax)
    plt.plot(cR_sec_resample, RRI_resample)
    plt.title('RRI_resampled')
    plt.ylabel('Hz')
    plt.subplot(414, sharex=ax)
    plt.plot(cR_sec, IFR)
    plt.title('IFR')
    plt.ylabel('Hz')
    #plt.show()

    # in this plot one RRI point correspond to the difference value between the precedent RR
    # the first point of RRI is the median for plotting consideration

    return RRI, RRI_resample, IFR, fig
    


#### LF / HF

#RRI_resample, srate_resample, nwind, nfft, noverlap, win = RRI_resample, srate_resample, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv
def get_PSD_LF_HF(RRI_resample, srate_resample, nwind, nfft, noverlap, win) :

    # DETREND
    RRI_detrend = RRI_resample-np.median(RRI_resample)

    # FFT WELCH
    hzPxx, Pxx = scipy.signal.welch(RRI_detrend, fs=srate_resample, window=win, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    # PLOT
    VLF, LF, HF = .05, .79, 3
    fig = plt.figure()
    plt.plot(hzPxx,Pxx)
    plt.ylim(0, np.max(Pxx[hzPxx>0.01]))
    plt.xlim([0,.6])
    plt.vlines([VLF, LF, HF], ymin=min(Pxx), ymax=max(Pxx))
    #plt.show()
    
    AUC_LF = np.trapz(Pxx[(hzPxx>VLF) & (hzPxx<LF)])
    AUC_HF = np.trapz(Pxx[(hzPxx>LF) & (hzPxx<HF)])
    LF_HF_ratio = AUC_LF/AUC_HF

    return AUC_LF, AUC_HF, LF_HF_ratio, fig, hzPxx, Pxx



#### SDNN, RMSSD, NN50, pNN50
# RR_val = RRI
def get_stats_descriptors(RR_val) :
    SDNN = np.std(RR_val)

    RMSSD = np.sqrt(np.mean((np.diff(RR_val)*1e3)**2))

    NN50 = []
    for RR in range(len(RR_val)) :
        if RR == len(RR_val)-1 :
            continue
        else :
            NN = abs(RR_val[RR+1] - RR_val[RR])
            NN50.append(NN)

    NN50 = np.array(NN50)*1e3
    pNN50 = np.sum(NN50>50)/len(NN50)

    return SDNN, RMSSD, NN50, pNN50

#SDNN_CV, RMSSD_CV, NN50_CV, pNN50_CV = get_stats_descriptors(RRI_CV)


#### Poincarré

def get_poincarre(RRI):
    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    fig = plt.figure()
    plt.scatter(RRI, RRI_1)
    plt.xlabel('RR (ms)')
    plt.ylabel('RR+1 (ms)')
    plt.title('Poincarré ')
    plt.xlim(.600,1.)
    plt.ylim(.600,1.)

    SD1_val = []
    SD2_val = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            SD1_val_tmp = (RRI[RR+1] - RRI[RR])/np.sqrt(2)
            SD2_val_tmp = (RRI[RR+1] + RRI[RR])/np.sqrt(2)
            SD1_val.append(SD1_val_tmp)
            SD2_val.append(SD2_val_tmp)

    SD1 = np.std(SD1_val)
    SD2 = np.std(SD2_val)
    Tot_HRV = SD1*SD2*np.pi

    return SD1, SD2, Tot_HRV, fig

    
#### DeltaHR

#RRI, srate_resample, f_RRI, condition = result_struct[keys_result[0]][1], srate_resample, f_RRI, cond 
def get_dHR(RRI_resample, srate_resample, f_RRI):
    
    times = np.arange(0,len(RRI_resample))/srate_resample

        # stairs method
    #RRI_stairs = np.array([])
    #len_cR = len(cR) 
    #for RR in range(len(cR)) :
    #    if RR == 0 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1]))])
    #    elif RR != 0 and RR != len_cR-1 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1] - cR[RR]))])
    #    elif RR == len_cR-1 :
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(len(ecg) - cR[RR]))])


    peaks, troughs = find_extrema(RRI_resample, srate_resample, f_RRI)
    peaks_RRI, troughs_RRI = RRI_resample[peaks], RRI_resample[troughs]
    peaks_troughs = np.stack((peaks_RRI, troughs_RRI), axis=1)

    fig_verif = plt.figure()
    plt.plot(times, RRI_resample)
    plt.vlines(peaks/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='b')
    plt.vlines(troughs/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='r')
    #plt.show()

    dHR = np.diff(peaks_troughs/srate_resample, axis=1)*1e3

    fig_dHR = plt.figure()
    ax = plt.subplot(211)
    plt.plot(times, RRI_resample*1e3)
    plt.title('RRI')
    plt.ylabel('ms')
    plt.subplot(212, sharex=ax)
    plt.plot(troughs/srate_resample, dHR)
    plt.hlines(np.median(dHR), xmin=min(times), xmax=max(times), colors='m', label='median = {:.3f}'.format(np.median(dHR)))
    plt.legend()
    plt.title('dHR')
    plt.ylabel('ms')
    plt.vlines(peaks/srate_resample, ymin=0, ymax=0.01, colors='b')
    plt.vlines(troughs/srate_resample, ymin=0, ymax=0.01, colors='r')
    plt.tight_layout()
    #plt.show()


    return fig_verif, fig_dHR


def ecg_analysis_homemade(ecg_i):

    #### load cR
    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srates['ecg']/15, threshold=0.02)[0]
    ecg_cR = ecg_cR/srates['ecg']

    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']

    #### RRI
    RRI, RRI_resample, IFR, fig_RRI = get_RRI_IFR(ecg_i, ecg_cR, srates['ecg'], srate_resample_hrv)

    HRV_MeanNN = np.mean(RRI)
    
    #### PSD
    AUC_LF, AUC_HF, LF_HF_ratio, fig_PSD, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, srate_resample_hrv, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv)

    #### descriptors
    SDNN, RMSSD, NN50, pNN50 = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV, fig_poincarre = get_poincarre(RRI)

    #### dHR
    fig_verif, fig_dHR = get_dHR(RRI_resample, srate_resample_hrv, f_RRI)

    #### fig
    fig_list = [fig_RRI, fig_PSD, fig_poincarre, fig_verif, fig_dHR]

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, AUC_LF, AUC_HF, SD1*1e3, SD2*1e3]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)

    return hrv_metrics_homemade, fig_list




########################################
######## ECG ANALYSIS NK ########
########################################

#rat, odeur, trial_i, ecg_i = 'B2', 'TriG', 'trial_1', data_rest['B2']['TriG']['ecg']['trial_1']
def ecg_analysis(rat, odeur, trial_i, ecg_i):

    #### load cR
    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srates['ecg']/15, threshold=0.02)[0]
    peaks_dict = {'ECG_R_Peaks' : ecg_cR}
    ecg_peaks = pd.DataFrame(peaks_dict)

    #### verif trig
    if debug:
        plt.plot(ecg_i)
        plt.vlines(ecg_cR, ymin=np.min(ecg_i), ymax=np.max(ecg_i), colors='r')
        plt.show()

    #### compute metrics
    hrv_metrics = nk.hrv(ecg_peaks, sampling_rate=srates['ecg'], show=False)

    total_hrv = np.array([hrv_metrics.iloc[0]['HRV_HF'] + hrv_metrics.iloc[0]['HRV_LF']])
    
    #### export 
    total_hrv = pd.DataFrame(data=total_hrv, index=None, columns=['HRV_Total'])

    hrv_metrics = pd.concat([hrv_metrics,total_hrv],axis=1)
    hrv_metrics.insert(0,'Rat',[rat])
    hrv_metrics.insert(1,'Odeur',[odeur])
    hrv_metrics.insert(2,'Trial',[trial_i+1])

    col_to_drop = []
    col_hrv = list(hrv_metrics.columns.values) 
    for metric_name in col_hrv :
        if metric_name == 'Rat' or metric_name == 'Odeur' or metric_name == 'Trial' :
            continue
        elif (metric_name in hrv_metrics_short_name) == False :
            col_to_drop.append(metric_name)

    hrv_metrics_short = hrv_metrics.copy()
    hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

    return hrv_metrics_short



def generate_hrv_fig(df_hrv):

    #### see all rats
    os.chdir(path_savefig)
    #hrv_metric_i = 'HRV_RMSSD'
    for hrv_metric_i in hrv_metrics_short_name:

        sns.set_theme(style="whitegrid")

        g = sns.catplot(x="Trial", y=hrv_metric_i, hue="Odeur", col="Rat", capsize=.2, palette="YlGnBu_d", height=6, aspect=.75, kind="point", data=df_hrv)
        g.despine(left=True)
        g.savefig(f'rest_{hrv_metric_i}.jpeg')


    #### groupby all rats
    for hrv_metric_i in hrv_metrics_short_name:

        sns.set_theme(style="whitegrid")

        #df_hrv.groupby(['Trial', 'Odeur']).mean().reset_index()

        g = sns.catplot(x="Trial", y=hrv_metric_i, hue="Odeur", capsize=.2, palette="YlGnBu_d", height=6, aspect=.75, kind="point", data=df_hrv)
        g.despine(left=True)
        g.savefig(f'rest_{hrv_metric_i}_allrats.jpeg')











if __name__ == '__main__':



    ########################
    ######## VISU ########
    ########################

    data = {}

    for rat in rats:
        data[rat] = {}
        for odeur in odeurs:
            data[rat][odeur] = {}
            for sig_type in sig_types:
            
                time, sig = chunk_sig(odeur, rat, sig_type, 'rest')
                data[rat][odeur][sig_type] = {'sig' : sig, 'time' : time}

    mean_all = []
    std_all = []

    for rat in rats:
        for odeur in odeurs:
            ecg_i = data[rat][odeur]['ecg']['sig']

            ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srates['ecg']/15, threshold=0.02)[0]

            vals = plt.hist(np.diff(ecg_cR), bins='auto')  # arguments are passed to np.histogram
            mean = np.mean(np.diff(ecg_cR))
            std_up = mean + np.std(np.diff(ecg_cR))
            std_down = mean - np.std(np.diff(ecg_cR))
            plt.vlines(mean, ymin=np.min(vals[0]) ,ymax=np.max(vals[0]), colors='r')
            plt.vlines(std_up, ymin=np.min(vals[0]) ,ymax=np.max(vals[0]), colors='b')
            plt.vlines(std_down, ymin=np.min(vals[0]) ,ymax=np.max(vals[0]), colors='b')
            plt.title(rat + '_' + odeur)
            plt.show()

            mean_all.append(mean)
            std_all.append(np.std(np.diff(ecg_cR)))








    ########################################
    ######## ANALYZE ECG REST ########
    ########################################

    #### load data rest
    data_rest = {}

    for rat in rats:
        data_rest[rat] = {}
        for odeur in odeurs:
            data_rest[rat][odeur] = {}
            for sig_type in ['ecg']:
                data_rest[rat][odeur][sig_type] = {}
                time, sig = chunk_sig(odeur, rat, sig_type, 'rest')
                for trial_i in range(4):                    
                    t_start = int(trial_i*5*60*srates[sig_type])
                    t_stop = int(t_start + 5*60*srates[sig_type])
                    data_rest[rat][odeur][sig_type][f'trial_{trial_i}'] = sig[t_start:t_stop]

    for rat in rats:
        for odeur in odeurs:
            for sig_type in ['ecg']:
                for trial_i in range(4):    
                    sig = data_rest[rat][odeur][sig_type][f'trial_{trial_i}']
                    print(rat, odeur, trial_i, sig.shape[0]/500/60)      

    
    #### compute
    df_hrv = pd.DataFrame(data=[], index=None, columns=hrv_metrics_short_name)
    df_hrv.insert(0,'Rat',[])
    df_hrv.insert(1,'Odeur',[])
    df_hrv.insert(2,'Trial',[])

    for rat in rats:
        if rat == 'R1': 
            continue
        for odeur in odeurs:
            for sig_type in ['ecg']:
                for trial_i in range(4):
                    print(rat, odeur, trial_i)
                    hrv_metrics_short = ecg_analysis(rat, odeur, trial_i, data_rest[rat][odeur][sig_type][f'trial_{trial_i}'])
                    hrv_metrics_homemade, fig_list = ecg_analysis_homemade(data_rest[rat][odeur][sig_type][f'trial_{trial_i}'])

                    hrv_metrics_short['HRV_LF'] = hrv_metrics_homemade['HRV_LF']
                    hrv_metrics_short['HRV_HF'] = hrv_metrics_homemade['HRV_HF']
                    hrv_metrics_short['HRV_LFHF'] = hrv_metrics_homemade['HRV_LF'] / hrv_metrics_homemade['HRV_HF']

                    df_hrv = pd.concat([df_hrv, hrv_metrics_short])





    #### save
    os.chdir(os.path.join(path_save, 'hrv'))
    df_hrv.to_excel('rest_hrv_allrats.xlsx')
    


    ################################
    ######## GENERATE FIG ########
    ################################

    os.chdir(os.path.join(path_save, 'hrv'))
    df_hrv = pd.read_excel('rest_hrv_allrats.xlsx')

    generate_hrv_fig(df_hrv)














