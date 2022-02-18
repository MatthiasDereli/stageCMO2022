


import numpy as np
import pandas as pd
import xarray as xr
import pingouin as pg
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from n0_config import *
from n0bis_analysis_functions import *

debug = False



def which_pre_test(df, dv, grouping):

    df = df.reset_index()

    normalities = pg.normality(data = df , dv = dv, group = grouping)['normal']
    
    if sum(normalities) == normalities.size:
        normality = True
    else:
        normality = False
        
    homoscedasticity = pg.homoscedasticity(data = df, dv = dv, group = grouping)['equal_var'].values[0]
    
    if normality and homoscedasticity:
        test_to_use = 'anova'
    else:
        test_to_use = 'friedman'

    return normality, test_to_use





def pre_and_post_hoc(df, within):
    
    p_values = {}
    rows_anov = []
    ttests = []
    
    for metric in df.columns:
        
        normality, test_to_use = which_pre_test(df=df, dv = metric , grouping=within)
        
        if test_to_use == 'anova':
            rm_anova = pg.rm_anova(data=df.reset_index(), dv = metric, within = within, subject = 'Rat')
            p_values[metric] = rm_anova.loc[:,'p-unc'].round(3).values[0]
            test_type = 'rm_anova'
            effsize = rm_anova.loc[:,'np2'].round(3).values[0]
        elif test_to_use == 'friedman':
            friedman = pg.friedman(data=df.reset_index(), dv = metric, within = within, subject = 'Rat')
            p_values[metric] = friedman.loc[:,'p-unc'].round(3).values[0]
            test_type = 'friedman'
            effsize = np.nan
            
        if p_values[metric] <= seuil : 
            significativity = 1
        else:
            significativity = 0
               
        row_anov = [metric , test_type , p_values[metric] , significativity, effsize]
        rows_anov.append(row_anov)
        
        ttest_metric = pg.pairwise_ttests(data=df.reset_index(), dv=metric, within=within, subject='Rat', parametric = normality, return_desc=True)
        ttest_metric.insert(0, 'metric', metric)
        ttests.append(ttest_metric)
        
    post_hocs = pd.concat(ttests)
    
    colnames = ['metric','test_type','pval', 'signif', 'effsize']
    df_pre = pd.DataFrame(rows_anov, columns = colnames)   

    return df_pre, post_hocs


def test_raw_to_signif(df_pre, post_hocs):
    mask = df_pre['signif'] == 1
    pre_signif = df_pre[mask]

    post_hocs_signif = post_hocs[post_hocs['p-unc'] < seuil]

    return pre_signif, post_hocs_signif



def post_hoc_interpretation(post_hocs_signif):
    

    conclusions = []
    
    for line in range(post_hocs_signif.shape[0]):
        
        metric = post_hocs_signif.reset_index().loc[line,'metric']
        cond1 = post_hocs_signif.reset_index().loc[line,'A']
        cond2 = post_hocs_signif.reset_index().loc[line,'B']
        
        hedge = np.abs(post_hocs_signif.reset_index().loc[line,'hedges'])

        if hedge <= 0.2:
            intensite = 'faible'
        elif hedge <= 0.8 and hedge >= 0.2:
            intensite = 'moyen'
        elif hedge >= 0.8:
            intensite = 'fort' 
        
        meanA = post_hocs_signif.reset_index().loc[line,'mean(A)']
        meanB = post_hocs_signif.reset_index().loc[line,'mean(B)']
            
        if meanA > meanB:
            comparateur = 'supérieur(e)'
        elif meanA < meanB:
            comparateur = 'inférieur(e)'

        conclusions.append(f"{metric} mesuré(e) en {cond1} est {comparateur} à {metric} mesuré(e) en {cond2} (effet {intensite})")
            
    return conclusions


def smart_stats(df, within):
    
    df_pre, df_post_hocs = pre_and_post_hoc(df=df, within=within)
    pre_signif, post_hocs_signif = test_raw_to_signif(df_pre, df_post_hocs)

    if pre_signif.shape[0] == 0:
        print('Pas de différence en pre_hoc')
    else:
        print('DIFF + en pre_test')
        
    if post_hocs_signif.shape[0] == 0:
        print('Pas de différence en post_hoc')
        conclusions = None
    else:
        print('DIFF + en post_hoc')
        conclusions = post_hoc_interpretation(post_hocs_signif)

    return df_pre, pre_signif, df_post_hocs, post_hocs_signif, conclusions




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    
    seuil = 0.05

    #### condition rest
    df = pd.read_excel(f'{root}cmo/Etudiants/NBuonviso202201_trigeminal_sna_rat_Mathias/Analyses/hrv/rest_hrv_allrats.xlsx')
    df = df.drop(columns = 'Unnamed: 0')
    indexes = ['Rat','Odeur','Trial']
    df = df.set_index(indexes)

    os.chdir(os.path.join(path_save, 'hrv'))
    df_pre, pre_signif, df_post_hocs, post_hocs_signif, conclusions = smart_stats(df, within = 'Trial')
    df_pre.to_excel('rest_stats_pre_within_trial.xlsx')
    df_post_hocs.to_excel('rest_stats_post_within_trial.xlsx')
    df_pre, pre_signif, df_post_hocs, post_hocs_signif, conclusions = smart_stats(df, within = 'Odeur')
    df_pre.to_excel('rest_stats_pre_within_odeur.xlsx')
    df_post_hocs.to_excel('rest_stats_post_within_odeur.xlsx')


    #### condition stress
    df = pd.read_excel(f'{root}cmo/Etudiants/NBuonviso202201_trigeminal_sna_rat_Mathias/Analyses/hrv/stress_hrv_allrats.xlsx')
    df = df.drop(columns = 'Unnamed: 0')
    indexes = ['Rat','Odeur','Stim']
    df = df.set_index(indexes)

    os.chdir(os.path.join(path_save, 'hrv'))
    df_pre, pre_signif, df_post_hocs, post_hocs_signif, conclusions = smart_stats(df, within = 'Stim')
    df_pre.to_excel('stress_stats_pre_within_stim.xlsx')
    df_post_hocs.to_excel('stress_stats_post_within_stim.xlsx')
    df_pre, pre_signif, df_post_hocs, post_hocs_signif, conclusions = smart_stats(df, within = 'Odeur')
    df_pre.to_excel('stress_stats_pre_within_odeur.xlsx')
    df_post_hocs.to_excel('stress_stats_post_within_odeur.xlsx')



