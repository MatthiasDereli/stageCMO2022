

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import h5py
import os
import neurokit2 as nk
import pandas as pd
import seaborn as sns

from n0_config import *
from n0bis_analysis_functions import *

debug = False




def generate_hrv_fig(df_stim_all):

    #### see all rats
    os.chdir(path_savefig)

    sns.set_theme(style="whitegrid")

    g = sns.catplot(x="Stim", y='HRV_MeanNN', hue="Odeur", col="Rat", capsize=.2, palette="YlGnBu_d", height=6, aspect=.75, kind="point", data=df_stim_all)
    g.despine(left=True)

    #plt.show()

    g.savefig(f'stress_HRV_MeanNN.jpeg')

    #### mean
    sns.set_theme(style="whitegrid")

    g = sns.catplot(x="Stim", y='HRV_MeanNN', hue="Odeur", capsize=.2, palette="YlGnBu_d", height=6, aspect=.75, kind="point", data=df_stim_all)
    g.despine(left=True)

    #plt.show()

    g.savefig(f'stress_HRV_MeanNN_allrats.jpeg')





#rat, odeur = 'N1', 'TriG'
def get_meanNN(rat, odeur, noise_stim_allrats):
    
    noise_stim_i = [int(i*srates['ecg']) for i in noise_stim_allrats[rat][odeur]]
    noise_stim_i = noise_stim_i[:n_noise_stim]
    ecg_i = data_stress_ecg[rat][odeur]['ecg']['signal']
    
    #### verif
    if debug:
        plt.plot(ecg_i)
        plt.vlines(noise_stim_i, ymin=np.min(ecg_i), ymax=np.max(ecg_i), colors='r')
        plt.title(f'{rat}_{odeur}')
        plt.show()

    df_stim = pd.DataFrame(columns=['Rat', 'Odeur', 'Stim', 'HRV_MeanNN'])

    for i, stim_i in enumerate(noise_stim_i):
        stim_sig = ecg_i[stim_i:stim_i+stress_time_to_take*srates['ecg']]
        ecg_cR = scipy.signal.find_peaks(stim_sig, distance=srates['ecg']/15, threshold=0.02)[0]

        if debug:
            plt.plot(stim_sig)
            plt.vlines(ecg_cR, ymin=np.min(stim_sig), ymax=np.max(stim_sig), colors='r')
            plt.title(f'{rat}_{odeur}')
            plt.show()

        #### compute metrics
        hrv_metrics = nk.hrv(ecg_cR, sampling_rate=srates['ecg'], show=False)

        total_hrv = np.array([hrv_metrics.iloc[0]['HRV_HF'] + hrv_metrics.iloc[0]['HRV_LF']])

        #### export 
        total_hrv = pd.DataFrame(data=total_hrv, index=None, columns=['HRV_Total'])

        hrv_metrics = pd.concat([hrv_metrics,total_hrv],axis=1)
        hrv_metrics.insert(0,'Rat',[rat])
        hrv_metrics.insert(1,'Odeur',[odeur])
        hrv_metrics.insert(2,'Stim',[f'{i+1}'])

        col_to_drop = []
        col_hrv = list(hrv_metrics.columns.values) 
        for metric_name in col_hrv :
            if metric_name == 'Rat' or metric_name == 'Odeur' or metric_name == 'Stim' or metric_name == 'HRV_MeanNN':
                continue
            else:
                col_to_drop.append(metric_name)

        hrv_metrics_short = hrv_metrics.copy()
        hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

        df_stim = pd.concat([df_stim, hrv_metrics_short])

    return df_stim






################################
######## EXECUTE ########
################################

if __name__ == '__main__':


    ################################
    ######## IDENTIFY STIM ########
    ################################

    #### load data stress
    data_stress = {}

    for rat in rats:
        data_stress[rat] = {}
        for odeur in odeurs:
            data_stress[rat][odeur] = {}
            for sig_type in sig_types:
                data_stress[rat][odeur][sig_type] = {}
                time, sig = chunk_sig(odeur, rat, sig_type, 'stress')
                data_stress[rat][odeur][sig_type]['time'] = time
                data_stress[rat][odeur][sig_type]['signal'] =  (sig - np.mean(sig))/np.std(sig)

    data_stress_ecg = {}

    for rat in rats:
        data_stress_ecg[rat] = {}
        for odeur in odeurs:
            data_stress_ecg[rat][odeur] = {}
            for sig_type in ['ecg']:
                data_stress_ecg[rat][odeur][sig_type] = {}
                time, sig = chunk_sig(odeur, rat, sig_type, 'stress')
                data_stress_ecg[rat][odeur][sig_type]['time'] = time
                data_stress_ecg[rat][odeur][sig_type]['signal'] =  sig

    #### verify
    if debug:
        for rat in rats:
            for odeur in odeurs:
                for sig_type in sig_types:
                    plt.plot(data_stress[rat][odeur][sig_type]['time'], data_stress[rat][odeur][sig_type]['signal'], label=sig_type)
                plt.legend()
                plt.title(f'{rat}_{odeur}')
                plt.show()


        #### select rats
        #['B1','B2','N1','N2','R1','R2','V1','V2']
        rat = 'B2'
        odeur = 'TriG'

        for sig_type in sig_types:
            plt.plot(data_stress[rat][odeur][sig_type]['time'], data_stress[rat][odeur][sig_type]['signal'], label=sig_type)
        plt.legend()
        plt.title(f'{rat}_{odeur}')
        plt.show()

        #### select trig
        noise_trig = []
        
        sig_scales = {'max' : [], 'min' : []} 
        for sig_type in sig_types:
            plt.plot(data_stress[rat][odeur][sig_type]['time'], data_stress[rat][odeur][sig_type]['signal'], label=sig_type)
            sig_scales['max'].append(np.max(data_stress[rat][odeur][sig_type]['signal']))
            sig_scales['min'].append(np.min(data_stress[rat][odeur][sig_type]['signal']))
        plt.vlines(noise_trig, ymin=np.min(sig_scales['min']), ymax=np.max(sig_scales['max']), colors='r')
        plt.legend()
        plt.title(f'{rat}_{odeur}')
        plt.show()

    

    #### fill rats values
    noise_stim_allrats = {}
    for rat in rats: 
        #### fill rats values
        if rat == 'B1':
            noise_stim = {'TriG' : [563.816, 608.019, 667.516, 810.312, 913.395, 1035.13, 1102.52, 1177.23, 1233.75, 1320.49, 1474.36, 1570.67, 1625.03, 1657.63],
            'NonTriG' : [542.347, 602.274, 662.467, 740.473, 791.411, 843.467, 901.467, 901.581, 963.415, 1023.24, 1086.03, 1144.86, 1204.75, 1266.25, 1331.12, 1383.71, 1439.71, 1510.06, 1565.67, 1626.75, 1684.72, 1743.7, 1803.01]
            }

        if rat == 'B2':
            noise_stim = {'TriG' : [541.741, 603.757, 664.506, 722.375, 782.374, 841.501, 900.565, 967.983, 1021.38, 1081.4, 1141.2, 1202.61, 1263.17, 1326.17, 1381.38, 1441.73, 1506.59, 1564.19, 1623.29, 1681.28],
            'NonTriG' : [542.262, 602.905, 664.118, 725.604, 787.367, 843.58, 901.842, 962.182, 1023.96, 1084.91, 1144.81, 1202.71, 1263.13, 1325.19, 1386.58, 1441.11, 1496.69, 1565.27, 1624.08, 1745.24, 1805.01, 1876.48, 1925.8, 1990.56, 2049.88, 2106.37, 2163.58, 2233.63, 2288.9]
            }

        if rat == 'N1':
            noise_stim = {'TriG' : [4.456, 120.552, 245.332, 365.536, 427.81, 490.352, 545.328, 605.102, 696.664, 777.65, 859.589, 932.094, 996.334,1085.86, 1132.91, 1265.68, 1338.63, 1510.04, 1676.21],
            'NonTriG' : [541.923, 571.036, 631.256, 721.44, 781.452, 841.114, 901.042, 961.456, 1011.14, 1083.23, 1141.51, 1202.3, 1259.43, 1321.5, 1383.42, 1442.02, 1561.61, 1624.95, 1684.38, 1743.51, 1803.59, 1863.61, 1922.24, 1981.55, 2042.75, 2102.08, 2161.51, 2223.33, 2283.78, 2349.83, 2403.41]
            }

        if rat == 'N2':
            noise_stim = {'TriG' : [542.29, 601.472, 662.348, 721.68, 781.946, 850.008, 902.846, 962.306, 1021.45, 1079.23, 1142.64, 1208.31, 1261.23, 1321.7, 1378.81, 1441.97, 1511.79, 1565.64, 1622.86, 1686.18, 1743.38, 1802.21],
            'NonTriG' : [601.146, 662.034, 721.966, 782.716, 822.356, 902.724, 961.242, 1021.43, 1085.08, 1142.71, 1202.27, 1265.87, 1324.15, 1394.05, 1446.46, 1501.6, 1623.63, 1683.54, 1743.96, 1815.12, 1865.24, 1923.76]
            }

        if rat == 'R1':
            noise_stim = {'TriG' : [],
            'NonTriG' : []
            }

        if rat == 'R2':
            noise_stim = {'TriG' : [543.368, 603.11, 662.376, 724.112, 778.433, 841.442, 901.74, 959.848, 1017.51, 1084.66, 1140.98, 1204.1, 1261.71, 1317.96, 1381.34, 1441.49, 1501.4, 1562.09, 1623.12, 1682.55, 1746.72, 1801.82, 1862.69, 1931.21, 1981.04, 2043.41, 2103.32, 2165.9],
            'NonTriG' : [541.882, 603.884, 661.39, 1023.18, 1086.88, 1141.89, 1214.15, 1274.97, 1364.78, 1449.82, 1511.91, 1570.51, 1640.24, 1706.15, 1754.94, 1821.87, 1867.53, 1947.37, 2004.93, 2066.22, 2132.96, 2194.96, 2261.98, 2299.4, 2329.62, 2356.02, 2395.92]
            }

        if rat == 'V1':
            noise_stim = {'TriG' : [451.507, 515.412, 578.214, 818.81, 909.626, 975.49, 1057.6, 1114.71, 1186.64, 1305.87, 1413.08, 1540.17, 1540.17, 1623.34, 1685.05, 1748.55, 1807.35, 1857.19],
            'NonTriG' : [618.243, 667.318, 726.769, 786.473, 844.83, 905.424, 969.97, 1032, 1093.46, 1146.88, 1205.37, 1265.06, 1325.44, 1385.02, 1447.62, 1507.25, 1568.58, 1627.11, 1685.51, 1755.9]
            }

        if rat == 'V2':
            noise_stim = {'TriG' : [542.146, 612.462, 680.242, 721.306, 781.386, 843.034, 901.582, 961.23, 1015.14, 1075.31, 1135.39, 1196.08, 1256.98, 1298.75, 1354.54, 1434.54, 1496.6, 1558.2, 1617.78],
            'NonTriG' : [241.542, 303.128, 365.086, 422.786, 503.104, 595.646, 658.514, 739.712, 831.098, 891.478, 977.738, 1039.3, 1083.03, 1136.76, 1193.64, 1242.6, 1317.97, 1383.67, 1441.69, 1496.99, 1560.52, 1628.18, 1675.66, 1721.11, 1765.25, 1800.54, 1838.01, 1876.72, 1902.66]
            }

        noise_stim_allrats[rat] = noise_stim


        

    ########################################
    ######## COMPUTE METRICS ########
    ########################################

    df_stim_all = pd.DataFrame(columns=['Rat', 'Odeur', 'Stim', 'HRV_MeanNN'])

    for rat in rats:
        for odeur in odeurs: 
            df_stim = get_meanNN(rat, odeur, noise_stim_allrats)
            df_stim_all = pd.concat([df_stim_all, df_stim])


    os.chdir(os.path.join(path_save, 'hrv'))
    df_stim_all.to_excel('stress_hrv_allrats.xlsx')

    ########################################
    ######## GENERATE FIGURES ########
    ########################################

    os.chdir(os.path.join(path_save, 'hrv'))
    df_stim_all = pd.read_excel('stress_hrv_allrats.xlsx')

    generate_hrv_fig(df_stim_all)



