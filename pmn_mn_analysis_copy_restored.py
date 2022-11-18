#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:28:04 2022

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
from scipy.stats import levene

#%% make plotting functions to fix all the problems with updating same figure
#heatmap connectivity matrices
def plot_p2m_weights(mat_pm,muscles,allmuscs,names,ti):
    f,ax = plt.subplots()
    sb.heatmap(mat_pm)
    #option: sb.clustermap?
    ax.set(xlabel="Muscles (D-->V)", ylabel="PMNs", yticks=np.arange(len(names)), yticklabels=names, xticks = muscles, xticklabels = allmuscs, title = ti)
    #plt.xticks(fontsize=10, rotation=0)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)

    plt.tight_layout()
    plt.show()

#simplified sum of PMN inputs to D L or V muscles
def plot_p2dlv_weights(mat_pd,grpnames,names):
    f,ax = plt.subplots()
    sb.heatmap(mat_pd)
    #option: sb.clustermap?
    ax.set(xlabel="Muscle Groups", ylabel="PMNs", yticks=np.arange(len(names)), yticklabels=names, xticklabels = grpnames, title = 'PMN Weights to General Muscle Groups')
    
    plt.tight_layout()
    plt.show()

#circum plot
def plot_musc_syndist(p2m,muscles,allmuscs,pnames):
    f,ax = plt.subplots()
    ax.barh(muscles, np.flip(allPMNs[p2m,:]))
    ax.set(xlabel="Synaptic Weights", ylabel="Muscles (V --> D)", yticks = muscles, yticklabels = allmuscs[::-1], ylim = (1,30), title = pnames[p2m])
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    
    plt.tight_layout()
    plt.show()
    
#PMN vs. wavg plot
def plot_xbar(wavgdist, shufsamp, regline, reglineshuf, ti):
    f,ax = plt.subplots()
    plt.plot(wavgdist, color = "blue", label="connectome")
    ax.set(xlabel="PMNs", ylabel="Weighted Average onto MNs",title = ti)
    for sa in np.arange(0,len(shufsamp)):
        plt.plot(shufsamp[:], color = "orange", alpha=0.3)
    plt.plot(regline, color = "black", label="connectome fit")
    plt.plot(reglineshuf, color = "red", label="shuffle fit")
    

#real vs. shuf variance measures boxplot
def plot_var(sigmas,grpnames):
    f,ax = plt.subplots()
    plt.boxplot(sigmas, grpnames)
    plt.xticks(ticks = [1,2], labels = grpnames)
    ax.set(xlabel="Real vs. Shuffled Connectivity", ylabel="Variance of PMN outputs")
    
#cosine similarity matrices
def plot_cos(cosmat,muscs,ti):
    f,ax = plt.subplots()
    sb.heatmap(cosmat, vmin=0, vmax=1)
    ax.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs,  yticks=np.arange(len(muscs)), yticklabels = muscs,title = ti)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        
#%% Load the data files with connectivity and NTs
Jpm = pd.read_excel('PMN to MN connectivity-matrix_correctroll.xlsx')
#Jpp = pd.read_excel('')
types = pd.read_excel('NT data 01 June.xlsx')

#%% Categorize the NT values
eNTs = ['Chat']
iNTs = ['Glut','GABA']

NTs = pd.Series(types.iloc[:,1])
NTvals = np.zeros(len(types))

einds = np.array(types.index[types['Chat'].isin(eNTs)])
iinds = np.array(types.index[types['Chat'].isin(iNTs)])

NTvals[einds] = 1
NTvals[iinds] = -1
#keep unknowns neutral as 0 and exclude from divided matrices
NTvals = NTvals[::2]

#%% Fix the naming issues
oldnames = np.array(types['A01c1_a1l'])
pnames = list()
ex_pnames = list()
in_pnames = list()

for ind,temp in enumerate(oldnames):
    if(ind%2) == 0:
        endname = temp.index('_')
        pnames.append(temp[:endname].lower().strip())
        if NTvals[int(ind/2)] == 1:
            ex_pnames.append(temp[:endname].lower().strip())
        elif NTvals[int(ind/2)] == -1:
            in_pnames.append(temp[:endname].lower().strip())

#%% DLV grouping
dor = ['1','9','2','10']
lat = ['11','19','3','18','20','4','24','23','22','21','8','5']
ven = ['12','13','30','14','6','7','25','26','27','29','28','15','16','17']
allmuscs = dor + lat + ven

#%%
#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epind = 0
ipind = 0

allPMNs = np.zeros([int(Jpm.shape[0]/2), len(allmuscs)])
ePMNs = np.zeros([len(ex_pnames), len(allmuscs)])
iPMNs = np.zeros([len(in_pnames), len(allmuscs)])

for pindie in np.arange(0,len(Jpm),2):
    #take sum of the two PMN-MN rows for each column
    lrpm = np.sum(np.array(Jpm.iloc[pindie:pindie+2,1:]),0).T
    for mi in np.arange(1,Jpm.shape[1],2):
        mind = list()
        #then take average of every two columns (L & R MNs) for each LR PMN pair
        pm = np.mean(lrpm[mi:mi+2])
        #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
        mtemp = Jpm.columns[mi+1]
        mtemp = mtemp.split('MN')[1]
        mtemp = mtemp[:mtemp.index(" ")]
        if '-' in mtemp:
            mtemp = mtemp[:mtemp.index("-")]
        #if '/' = 2 MNs, then store for the number strings before and after the '/'
        if '/' in mtemp:
            mtemp = mtemp.split('/')
            for im,mn in enumerate(mtemp):
                mind.append(allmuscs.index(mtemp[im])) #find idx of this mn in the allmuscs list
        else:
             mind = allmuscs.index(mtemp)   
        #store in PMN row MN col
        pind = int(pindie/2)
        allPMNs[pind,mind] = pm
        #print('a_' + str(pindie) + '_' + str(pind) + ',' + str(mind))
        
        #break into E vs. I matrices
        if NTvals[int(pindie/2)] == 1:
         #   print('e_' + str(pindie) + '_' + str(epind) + ',' + str(mind))
            ePMNs[epind,mind] = pm
            if mind == 18:
                epind = epind + 1
            
        elif NTvals[int(pindie/2)] == -1:
          #  print('i_' + str(pindie) + '_' + str(ipind) + ',' + str(mind))
            iPMNs[ipind,mind] = pm
            if mind == 18:
                ipind = ipind + 1
            

#%% PMN DLV group plots
#plot grouped weights matrices
muscles = np.arange(0,30)
plot_p2m_weights(allPMNs,muscles,allmuscs,pnames, 'All PMNs - spatial muscle order')
plot_p2m_weights(ePMNs,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - spatial muscle order')
plot_p2m_weights(iPMNs,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - spatial muscle order')



#%% weighted avg for each PMN according to DLV grps, sort, and replot
#fxn for generating and sorting PMNs according to weighted DLV sums

# 1. assign locations 1-30 - muscles
# 2. take weighted average for literally that number for each PMN
# 3. sort PMN rows by where weighted average is highest for that PMN
def wavg(mat_syn, muscs):
    mat_out = np.zeros(len(mat_syn))
    var_out = np.zeros(len(mat_syn))
    
    for pm in np.arange(0,len(mat_syn)):
        pmint = int(pm)
        mat_out[pmint] = np.average(muscs, weights = mat_syn[pmint,:])
        var_out[pmint] = np.average((muscs - mat_out[pmint])**2, weights = mat_syn[pmint,:])
        
    sortpmns = mat_out.argsort()
    reordp = mat_syn[sortpmns,:]
    zreord = stats.zscore(reordp, axis = 1)
    
    xj = mat_out[sortpmns]
    
    #mat_out, sortpmns, reordp, zreord,
    
    return  xj, var_out, reordp, zreord
#%%
xjall, sigjall, reordall, zreall = wavg(allPMNs, muscles)
xjex, sigjex, reordex, zreex  = wavg(ePMNs, muscles)
xjin, sigjin, reordin, zrein = wavg(iPMNs, muscles)

#%%
#plot the sorted PMNs
plot_p2m_weights(reordall,muscles,allmuscs,pnames, 'All PMNs - reordered by weighted average')
plot_p2m_weights(zreall,muscles,allmuscs,pnames, 'All PMNs - reordered by weighted average, z-score')

plot_p2m_weights(reordex,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - reordered by weighted average')
plot_p2m_weights(zreex,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - reordered by weighted average, z-score')

plot_p2m_weights(reordin,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - reordered by weighted average')
plot_p2m_weights(zrein,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - reordered by weighted average, z-score')

#%%
#do it again for the locations being spatially clustered?
# musclocs = np.flip(np.array([14, 14, 13, 13, 12.5, 12, 11, 11, 11, 10.5,
#                      10, 9, 9, 9, 8, 8, 6, 5.5, 5, 5, 4.5, 4.5,
#                      4, 3.5, 3.5, 3, 2.5, 2, 1.5, 1]));

# wavgall, centroids, reordall, zreall, xjalllocs, sigjalllocs = wavg(allPMNs, musclocs)
# wavgex, centex, reordex, zreex, xjexlocs, sigjexlocs = wavg(ePMNs, musclocs)
# wavgin, centin, reordin, zrein, xjinlocs, sigjinlocs = wavg(iPMNs, musclocs)

#did not make substantial difference - do not pursue

#%%
#plot the sorted PMNs
# plot_p2m_weights(reordall,muscles,allmuscs,pnames)
# plot_p2m_weights(zreall,muscles,allmuscs,pnames)

# plot_p2m_weights(reordex,muscles,allmuscs,ex_pnames)
# plot_p2m_weights(zreex,muscles,allmuscs,ex_pnames)

# plot_p2m_weights(reordin,muscles,allmuscs,in_pnames)
# plot_p2m_weights(zrein,muscles,allmuscs,in_pnames)

#%%plot PMNs vs. wavg
#first sort the wavg by order
# plot_xbar(xjall, pnames,'All PMNs - weighted avg distrib')
# plot_xbar(xjex, pnames, 'Excitatory PMNs - weighted avg distrib')
# plot_xbar(xjin, pnames, 'Inhibitory PMNs - weighted avg distrib')

# # plot_xbar(xjalllocs, pnames)
# # plot_xbar(xjexlocs, pnames)
# # plot_xbar(xjinlocs, pnames)

#%% fxn for the shuffle comparison
#shuffle weights matrix 1000x, choose PMN partners based on prob of MN input
def shufmat(cnxns,num_reps):
    rand_mats = [] #set this up to be a 1000d array; store each, then perform the wavg on each -- will extract mean xj or even plot all light then do avg dark; same with vars
    bicnxns = np.where(cnxns > 0, 1, 0)
    outputPMNs = np.sum(bicnxns,1) 
    totalMNin = np.sum(np.sum(bicnxns,0),0)
    inputMNs = (np.sum(bicnxns,0)) / totalMNin
    
    P = bicnxns.shape[0]
    M = bicnxns.shape[1]
    
    Wshuf = np.zeros([P,M])
    
    for rep in range(num_reps):
        for pout in range(P):
            outputs = np.random.choice(M, outputPMNs[pout], replace=False, p=inputMNs)
            Wshuf[pout,outputs] = 1
        rand_mats.append(Wshuf)
        Wshuf = np.zeros([P,M])
        
    return rand_mats
    

#%% shuffle matrices generation for all, e and i
randall = shufmat(allPMNs,1000)
randex = shufmat(ePMNs,1000)
randin = shufmat(iPMNs,1000)

#%%
#run the wavg fxn on the shuf mats
xjshufall = np.zeros([len(allPMNs),len(randall)])
xjshufex = np.zeros([len(ePMNs),len(randall)])
xjshufin = np.zeros([len(iPMNs),len(randall)])

varshufall = np.zeros([len(allPMNs),len(randall)])
varshufex = np.zeros([len(ePMNs),len(randall)])
varshufin = np.zeros([len(iPMNs),len(randall)])

matall = np.zeros([allPMNs.shape[0],allPMNs.shape[1],len(randall)])
matex = np.zeros([ePMNs.shape[0], ePMNs.shape[1],len(randall)])
matin = np.zeros([iPMNs.shape[0], iPMNs.shape[1],len(randall)])

zmatall = np.zeros([allPMNs.shape[0],allPMNs.shape[1],len(randall)])
zmatex = np.zeros([ePMNs.shape[0], ePMNs.shape[1],len(randall)])
zmatin = np.zeros([iPMNs.shape[0], iPMNs.shape[1],len(randall)])

for dim in np.arange(0,len(randall)):
    xjshufall[:,dim], varshufall[:,dim], matall[:,:,dim], zmatall[:,:,dim] = wavg(randall[dim], muscles)
    xjshufex[:,dim], varshufex[:,dim], matex[:,:,dim], zmatex[:,:,dim] = wavg(randex[dim], muscles)
    xjshufin[:,dim], varshufin[:,dim], matin[:,:,dim], zmatin[:,:,dim] = wavg(randin[dim], muscles)

#%% plot some of the shuffled matrices and see how they compare
pickrall = np.random.randint(0, matall.shape[2], size = 20)
pickrex = np.random.randint(0, matex.shape[2], size = 20)
pickrin = np.random.randint(0, matin.shape[2], size = 20)

# for i in np.arange(0,len(pickrall)):
#     plot_p2m_weights(matall[:,:,pickrall[i]],muscles,allmuscs,pnames,'PMN-MN Shuffled Weights - All - matrix '+ str(pickrall[i]))
#     plot_p2m_weights(matex[:,:,pickrex[i]],muscles,allmuscs,ex_pnames,'PMN-MN Shuffled Weights - Excitatory matrix'+ str(pickrex[i]))
#     plot_p2m_weights(matin[:,:,pickrin[i]],muscles,allmuscs,in_pnames,'PMN-MN Shuffled Weights - Inhibitory matrix'+ str(pickrin[i]))

#%% compare binary plots
biall = np.where(reordall>0, 1, 0)
biex = np.where(reordex>0, 1, 0)
biin = np.where(reordin>0, 1, 0)

plot_p2m_weights(biall,muscles,allmuscs,pnames,'PMN-MN Connectome Weights - All')
plot_p2m_weights(biex,muscles,allmuscs,ex_pnames,'PMN-MN Connectome Weights - Excitatory')
plot_p2m_weights(biin,muscles,allmuscs,in_pnames,'PMN-MN Shuffled Weights - Inhibitory')
# #%%
# #compare xj's
mxj_all = np.mean(xjshufall, axis = 1)
mxj_ex = np.mean(xjshufex, axis = 1)
mxj_in = np.mean(xjshufin, axis = 1)

# plot_xbar(xjall, mxj_all, pnames)
# plot_xbar(xjex, mxj_ex, pnames)
# plot_xbar(xjin, mxj_in, pnames)

# #%% compare xj's across 20x shuf vs connectome
# plot_xbar(xjall, xjshufall[:,pickrall], "All PMNs")
# plot_xbar(xjex, xjshufex[:,pickrex], "Excitatory PMNs")
# plot_xbar(xjin, xjshufin[:,pickrin], "Inhibitory PMNs")

#%%
#draw regression lines and check if regressions are significantly different
import statsmodels.api as sm
def regressline(sample_xbar):
    X = sm.add_constant(np.arange(0,len(sample_xbar)))
    model = sm.OLS(sample_xbar,X)
    results = model.fit()
    params = model.fit().params
    
    #regline = params * np.arange(0,len(sample_xbar))
    regline = (params[0] + params[1] * X[:,1])
    
    return regline, results

#define f test
def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-stats.f.cdf(f, nun, dun)
    return f, p_value

#regress and check if population variances of samples are equal, ftest for variances of populations
rall,mrall = regressline(xjall)
rshufall,mrshufall = regressline(mxj_all)
varallvshuf = levene(mrall.resid, mrshufall.resid)
fall = f_test(xjall,mxj_all)

rex,mrex = regressline(xjex)
rshufex,mrshufex = regressline(mxj_ex)
varexvshuf = levene(mrex.resid, mrshufex.resid)
fex = f_test(xjex,mxj_ex)

rin,mrin = regressline(xjin)
rshufin,mrshufin = regressline(mxj_in)
varinvshuf = levene(mrin.resid, mrshufin.resid)
fin = f_test(xjin,mxj_in)

#plot the data plus the regression lines
plot_xbar(xjall, xjshufall[:,pickrall], rall, rshufall, "All PMNs")
plot_xbar(xjex, xjshufex[:,pickrex], rex, rshufex, "Excitatory PMNs")
plot_xbar(xjin, xjshufin[:,pickrin], rin, rshufin, "Inhibitory PMNs")

#%%
#compare var's
# msigma_all = np.mean(varshufall, axis = 1)
# msigma_ex = np.mean(varshufex, axis = 1)
# msigma_in = np.mean(varshufin, axis = 1)

msigma_all = varshufall[:,20]
msigma_ex = varshufex[:,20]
msigma_in = varshufin[:,20]

allvall = np.column_stack([sigjall, msigma_all])
eve = np.column_stack([sigjex, msigma_ex])
ivi = np.column_stack([sigjin, msigma_in])

plot_var(allvall,['All PMNs','Shuf All'])
plot_var(eve,['Excit PMNs','Shuf Excit'])
plot_var(ivi,['Inhib PMNs','Shuf Inhib'])
    
#%% compare the variances of real PMN-MN spread vs. shuffled w/ t-test
tallvshufall = stats.ttest_ind(sigjall, msigma_all)
texvshufex = stats.ttest_ind(sigjex, msigma_ex)
tinvshufin = stats.ttest_ind(sigjin, msigma_in)

#shows that indiv PMN spread is sig more spatially localized in connectome data than shuffled data



#%% calculate the norm of D, L, and V matrices for connectome v shuffled data as a metric for overlap
#if product of norms = 0, then orthogonal, no overlap; if > 0, non-orthogonal
dorfxinds = np.arange(0,len(dorfx))
latfxinds = np.arange(len(dorfx),len(dorfx)+len(latfx))
venfxinds = np.arange(len(dorfx)+len(latfx),len(dorfx)+len(latfx)+len(venfx))

#despite rerord matrices looking similar, group this matrix comparison according to my new spatial + fxnal muscle groups --> dorfx, latfx, venfx
dorall = biallfx[:,dorfxinds]
tdorall = dorall.T
latall = biallfx[:,latfxinds]

testdl = tdorall @ latall
normtest = np.linalg.norm(testdl)
normtestdor = np.linalg.norm(tdorall)
normtestlat = np.linalg.norm(latall)

venall = biallfx[:,venfxinds]

norm_all_dor = np.linalg.norm(dorall)
norm_all_lat = np.linalg.norm(latall)
norm_all_ven = np.linalg.norm(venall)

#%%
#this makes no sense b/c of course these would be positive values? try instead
#do dot product of row vectors for PMNs dor, lat, ven
for td in tdorall:
    testdot = np.dot(tdorall[td,:],latall[:,td])
checkorthog = np.where(testdot==0)

# #do norm of row and col, then take product, sum all?
# dotdl = []
# for td in tdorall:
#     testnormd = np.linalg.norm(tdorall[td,:])
#     testnorml = np.linalg.norm(latall[:,td])
#     dotdl[td] = testnormd*testnorml

#another idea: try taking eigenvecs of each matrix and see if orthog
#can't do with non-square matrix and can't square this?
# eigdor = np.linalg.eig(dorall)
# eiglat = np.linalg.eig(latall)
# eigven = np.linalg.eig(venall)

#%% try cosine similarity comparisons of matrices -- shows sim of seq of numbers
#inner dot product of vectors divided by norm of vectors
#here, operate per PMN (colvecs) across MN grps (rows)
#so output is dorMNs to latMNs - 0 = orthog, 1 = overlap 100%

import scipy.spatial as sp

cosall = 1 - sp.distance.cdist(biallfx.T, biallfx.T, 'cosine')
cosex = 1 - sp.distance.cdist(biexfx.T, biexfx.T, 'cosine')
cosin = 1 - sp.distance.cdist(biinfx.T, biinfx.T, 'cosine')

#replot with muscle labels and appropriate titles
plot_cos(cosall, allmuscsfx, pnames,'Cosine Similarity of Connections- All PMNs')
plot_cos(cosex, allmuscsfx, ex_pnames,'Cosine Similarity of Connections - Excitatory PMNs')
plot_cos(cosin, allmuscsfx, in_pnames,'Cosine Similarity of Connections - Inhibitory PMNs')

#now repeat this but with some of the shuffled binary mats


#%%
#pull out the connectivity for these
def findsubsets(inds):
    newmatall = biallfx[:,inds]
    newmatex = biexfx[:,inds]
    newmatin = biinfx[:,inds]
    
    return newmatall, newmatex, newmatin
#%%
#pull out the connectivity for the shuffs
def findsubsetsshuf(inds,indmats):
    newshufall = np.zeros([len(matall),len(inds),len(pickrall)])
    newshufex = np.zeros([len(matex),len(inds),len(pickrex)])
    newshufin = np.zeros([len(matin),len(inds),len(pickrin)])
    
    for i in np.arange(0,len(pickrall)):
        newshufall = np.dstack((newshufall,matall[:,inds,indmats[i]])).shape
        newshufex = np.dstack((newshufex,matex[:,inds,indmats[i]])).shape
        newshufin = np.dstack((newshufin,matin[:,inds,indmats[i]])).shape
    
    return newshufall, newshufex, newshufin
#%%
#run cossim and store ACS
def cossim(compmats):
    acsmat = np.zeros([len(compmats),len(compmats)])
    for m in np.arange(0,len(compmats)):
        j = m + 1
        selfcos = np.nanmean(1 - sp.distance.cdist(compmats[m].T, compmats[m].T, 'cosine'))
        selfmat = 1 - sp.distance.cdist(compmats[m].T, compmats[m].T, 'cosine')
        acsmat[m,m] = selfcos
        while j < len(compmats):
            othercos = np.nanmean(1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine'))
            othermat = 1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine')
            acsmat[m,j] = othercos
            acsmat[j,m] = othercos
            j = j + 1
    return acsmat
#%%
#option: take average cosine similarity of groups DL to LL, etc. and plot distrib of this compared to the shuffled 1000x distrib to see if significant
#redo into smaller subgrps based on first plot by eye
# D1 (1, 9, 2, 10, 3, 11)
# D2L1 (19, 20, 18, 24, 23, 22, 21, 8)
# V2 (25, 26, 27, 29)
# L2V1 (4, 5, 12, 13, 30, 14)
# V3 (6, 7, 28, 15, 16, 17)

#previous 
# dorfx = ['1','9','2','10','3','11','19','20']
# latfx = ['18','24','23','22','21','8','25','26','27','29']
# venfx = ['4','5','12','13','30','14','6','7','28','15','16','17']

d1 = ['1','9','2','10','3','11']
d1inds = np.arange(0,len(d1))
d2l1 = ['19','20','18','24','23','22','21','8']
d2inds = len(d1inds) + np.arange(0,len(d2l1))
v2 = ['25','26','27','29']
v2inds =  len(d1inds) + len(d2inds) + np.arange(0,len(v2))
l2v1 = ['4','5','12','13','30','14']
l2v1inds = len(d1inds) + len(d2inds) + len(v2inds) + np.arange(0,len(l2v1))
v3 = ['6','7','28','15','16','17']
v3inds = len(d1inds) + len(d2inds) + len(l2v1inds) + len(v2inds) + np.arange(0,len(v3))
newmuscorder = d1 + d2l1 + v2 + l2v1 + v3

d1matall, d1matex, d1matin = findsubsets(d1inds)
d2l1matall, d2l1matex, d2l1matin = findsubsets(d2inds)
v2matall, v2matex, v2matin = findsubsets(v2inds)
l2v1matall, l2v1matex, l2v1matin = findsubsets(l2v1inds)
v3matall, v3matex, v3matin = findsubsets(v3inds)
#%%
#repeat for the pickr shuff subsets





##########STOPPED HERE - FIX THE SHUFFLED MATRIX SELECTIONS FOR MUSCLE SUBGRPS,
#THEN FIND ACS FOR EACH AND STORE
#THEN PLOT THE DISTRIBUTION OF ACS VALS FOR THE SHUFF MATS, AND PLOT THE ACS VALS FOR REAL DATA IN THE DISTRIB IN DIFF COLOR
#THEN DO STAT TEST (SEE PAPER AGAIN -- WILCOXON OR OTHER?)

















d1shufall, d1shufex, d1shufin = findsubsetsshuf(d1inds,pickrall)
d2l1shufall, d2l1shufex, d2l1shufin = findsubsetsshuf(d2inds,pickrall)
v2shufall, v2shufex, v2shufin = findsubsetsshuf(v2inds,pickrall)
l2v1shufall, l2v1shufex, l2v1shufin = findsubsetsshuf(l2v1inds,pickrall)
v3shufall, v3shufex, v3shufin = findsubsetsshuf(v3inds,pickrall)

#%%
#calculate groupwise cosine similarities - real data and for loop shuff mats
acsall = cossim([d1matall,d2l1matall,v2matall,l2v1matall,v3matall])
acsex = cossim([d1matex,d2l1matex,v2matex,l2v1matex,v3matex])
acsin = cossim([d1matin,d2l1matin,v2matin,l2v1matin,v3matin])

#plot the cosine similarity matrices per muscle group
groupnames = ['D1','D2/L1','V1','L2/V2','V3']
plot_cos(acsall, groupnames,'Cosine Similarity Across Muscle Groups- All PMNs')
plot_cos(acsex, groupnames,'Cosine Similarity Across Muscle Groups - Excitatory PMNs')
plot_cos(acsin, groupnames,'Cosine Similarity Across Muscle Groups - Inhibitory PMNs')

# for i in np.arange(0,len(pickrall)):
#     plot_p2m_weights(matall[:,:,pickrall[i]],

#plot the distrib of acs for each pairwise grouping - shufmats and the acs for realmats


















#%% do all again but with the new MN order based on DLV and fxnal groups
#%% DLV grouping - this time 4,5,12 later than LTs; can also try with them before
dorfx = ['1','9','2','10','3','11','19','20']
latfx = ['18','24','23','22','21','8','25','26','27','29']
venfx = ['4','5','12','13','30','14','6','7','28','15','16','17']
allmuscsfx = dorfx + latfx + venfx

#%% reorder connectome according to this muscle order
#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epind = 0
ipind = 0

allPMNsfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])
ePMNsfx = np.zeros([len(ex_pnames), len(allmuscsfx)])
iPMNsfx = np.zeros([len(in_pnames), len(allmuscsfx)])

for pindie in np.arange(0,len(Jpm),2):
    #take sum of the two PMN-MN rows for each column
    lrpm = np.sum(np.array(Jpm.iloc[pindie:pindie+2,1:]),0).T
    for mi in np.arange(1,Jpm.shape[1],2):
        mind = list()
        #then take average of every two columns (L & R MNs) for each LR PMN pair
        pm = np.mean(lrpm[mi:mi+2])
        #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
        mtemp = Jpm.columns[mi+1]
        mtemp = mtemp.split('MN')[1]
        mtemp = mtemp[:mtemp.index(" ")]
        if '-' in mtemp:
            mtemp = mtemp[:mtemp.index("-")]
        #if '/' = 2 MNs, then store for the number strings before and after the '/'
        if '/' in mtemp:
            mtemp = mtemp.split('/')
            for im,mn in enumerate(mtemp):
                mind.append(allmuscsfx.index(mtemp[im])) #find idx of this mn in the allmuscs list
        else:
              mind = allmuscsfx.index(mtemp)   
        #store in PMN row MN col
        pind = int(pindie/2)
        allPMNsfx[pind,mind] = pm
        print('a_' + str(pindie) + '_' + str(pind) + ',' + str(mind))
        
        #break into E vs. I matrices
        if NTvals[int(pindie/2)] == 1:
            print('e_' + str(pindie) + '_' + str(epind) + ',' + str(mind))
            ePMNsfx[epind,mind] = pm
            if mind == 22:
                epind = epind + 1
            
        elif NTvals[int(pindie/2)] == -1:
            print('i_' + str(pindie) + '_' + str(ipind) + ',' + str(mind))
            iPMNsfx[ipind,mind] = pm
            if mind == 22:
                ipind = ipind + 1
                
#%% PMN DLV group plots
#plot grouped weights matrices
muscles = np.arange(0,30)
plot_p2m_weights(allPMNsfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")
plot_p2m_weights(ePMNsfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")
plot_p2m_weights(iPMNsfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")

#%% weighted avg and sort and plot
xjallfx, sigjallfx, reordallfx, zreallfx = wavg(allPMNsfx, muscles)
xjexfx, sigjexfx, reordexfx, zreexfx = wavg(ePMNsfx, muscles)
xjinfx, sigjinfx, reordinfx, zreinfx = wavg(iPMNsfx, muscles)

plot_p2m_weights(reordallfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")
plot_p2m_weights(zreallfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")

plot_p2m_weights(reordexfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")
plot_p2m_weights(zreexfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")

plot_p2m_weights(reordinfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")
plot_p2m_weights(zreinfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")

#%% compare binary plots
biallfx = np.where(reordallfx>0, 1, 0)
biexfx = np.where(reordexfx>0, 1, 0)
biinfx = np.where(reordinfx>0, 1, 0)

plot_p2m_weights(biallfx,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - reorg muscs')
plot_p2m_weights(biexfx,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - reorg muscs')
plot_p2m_weights(biinfx,muscles,allmuscsfx,in_pnames,'PMN-MN Shuffled Weights - Inhibitory - reorg muscs')


#%% shuffle matrices generation for all, e and i
randall = shufmat(allPMNsfx,1000)
randex = shufmat(ePMNsfx,1000)
randin = shufmat(iPMNsfx,1000)

#%%
#run the wavg fxn on the shuf mats
xjshufall = np.zeros([len(allPMNsfx),len(randall)])
xjshufex = np.zeros([len(ePMNsfx),len(randall)])
xjshufin = np.zeros([len(iPMNsfx),len(randall)])

varshufall = np.zeros([len(allPMNsfx),len(randall)])
varshufex = np.zeros([len(ePMNsfx),len(randall)])
varshufin = np.zeros([len(iPMNsfx),len(randall)])

matall = np.zeros([allPMNsfx.shape[0],allPMNsfx.shape[1],len(randall)])
matex = np.zeros([ePMNsfx.shape[0], ePMNsfx.shape[1],len(randall)])
matin = np.zeros([iPMNsfx.shape[0], iPMNsfx.shape[1],len(randall)])

zmatall = np.zeros([allPMNsfx.shape[0],allPMNsfx.shape[1],len(randall)])
zmatex = np.zeros([ePMNsfx.shape[0], ePMNsfx.shape[1],len(randall)])
zmatin = np.zeros([iPMNsfx.shape[0], iPMNsfx.shape[1],len(randall)])

for dim in np.arange(0,len(randall)):
    xjshufall[:,dim], varshufall[:,dim], matall[:,:,dim], zmatall[:,:,dim] = wavg(randall[dim], muscles)
    xjshufex[:,dim], varshufex[:,dim], matex[:,:,dim], zmatex[:,:,dim] = wavg(randex[dim], muscles)
    xjshufin[:,dim], varshufin[:,dim], matin[:,:,dim], zmatin[:,:,dim] = wavg(randin[dim], muscles)

#%% plot some of the shuffled matrices and see how they compare
pickrall = np.random.randint(0, matall.shape[2], size = 20)
pickrex = np.random.randint(0, matex.shape[2], size = 20)
pickrin = np.random.randint(0, matin.shape[2], size = 20)
#%%
# #compare xj's
# mxj_all = np.mean(xjshufall, axis = 1)
# mxj_ex = np.mean(xjshufex, axis = 1)
# mxj_in = np.mean(xjshufin, axis = 1)

# plot_xbar(xjallfx, mxj_all, pnames)
# plot_xbar(xjexfx, mxj_ex, pnames)
# plot_xbar(xjinfx, mxj_in, pnames)

#%% compare xj's across 20x shuf vs connectome
# plot_xbar(xjallfx, xjshufall[:,pickrall], "All PMNs")
# plot_xbar(xjexfx, xjshufex[:,pickrex], "Excitatory PMNs")
# plot_xbar(xjinfx, xjshufin[:,pickrin], "Inhibitory PMNs")

#%%
#draw regression lines and check if regressions are significantly different
import statsmodels.api as sm
def regressline(sample_xbar):
    X = sm.add_constant(np.arange(0,len(sample_xbar)))
    model = sm.OLS(sample_xbar,X)
    results = model.fit()
    params = model.fit().params
    
    #regline = params * np.arange(0,len(sample_xbar))
    regline = (params[0] + params[1] * X[:,1])
    
    return regline, results

#define f test
def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-stats.f.cdf(f, nun, dun)
    return f, p_value

#regress and check if population variances of samples are equal, ftest for variances of populations
rall,mrall = regressline(xjallfx)
rshufall,mrshufall = regressline(mxj_all)
varallvshuf = levene(mrall.resid, mrshufall.resid)
fall = f_test(xjallfx,mxj_all)

rex,mrex = regressline(xjexfx)
rshufex,mrshufex = regressline(mxj_ex)
varexvshuf = levene(mrex.resid, mrshufex.resid)
fex = f_test(xjexfx,mxj_ex)

rin,mrin = regressline(xjinfx)
rshufin,mrshufin = regressline(mxj_in)
varinvshuf = levene(mrin.resid, mrshufin.resid)
fin = f_test(xjinfx,mxj_in)

#plot the data plus the regression lines
plot_xbar(xjallfx, xjshufall[:,pickrall], rall, rshufall, "All PMNs")
plot_xbar(xjexfx, xjshufex[:,pickrex], rex, rshufex, "Excitatory PMNs")
plot_xbar(xjinfx, xjshufin[:,pickrin], rin, rshufin, "Inhibitory PMNs")

#%%
#compare var's
# msigma_all = np.mean(varshufall, axis = 1)
# msigma_ex = np.mean(varshufex, axis = 1)
# msigma_in = np.mean(varshufin, axis = 1)

# msigma_all = varshufall[:,20]
# msigma_ex = varshufex[:,20]
# msigma_in = varshufin[:,20]

# allvall = np.column_stack([sigjall, msigma_all])
# eve = np.column_stack([sigjex, msigma_ex])
# ivi = np.column_stack([sigjin, msigma_in])

# plot_var(allvall,['All PMNs','Shuf All'])
# plot_var(eve,['Excit PMNs','Shuf Excit'])
# plot_var(ivi,['Inhib PMNs','Shuf Inhib'])
    
# #%% compare the variances of real PMN-MN spread vs. shuffled w/ t-test
# tallvshufall = stats.ttest_ind(sigjall, msigma_all)
# texvshufex = stats.ttest_ind(sigjex, msigma_ex)
# tinvshufin = stats.ttest_ind(sigjin, msigma_in)

#shows that indiv PMN spread is sig more spatially localized in connectome data than shuffled data

#%% not using anymore


# # #indiv circumferential connectivity check
# # muscles = np.arange(0,30)

# # #plot each PMN's connectivity distribution profile
# # for pi,pr in enumerate(allPMNs):
# #     plot_musc_syndist(pi,muscles,allmuscs,pnames)

# List generation and heatmap of PMN DLV groups
#function for simplified weights DLV
# def dlvsum(mat_in,dorinds,latinds,veninds):
#     mat_out = np.zeros([len(mat_in),3])
#     for i,p in enumerate(mat_in):
#         dorPMNs = sum(mat_in[i,dorinds])
#         latPMNs = sum(mat_in[i,latinds])
#         venPMNs = sum(mat_in[i,veninds])
#         mat_out[i,:] = [dorPMNs, latPMNs, venPMNs]
        
#     return mat_out

# #go through allPMNs and sum weights to dor, lat, and ven indices, and generate lists
# gnames = ['dorsal','lateral','ventral']
# dorinds = np.arange(0, len(dor))
# latinds = np.arange(len(dor), len(dor)+len(lat))
# veninds = np.arange(len(dor)+len(lat), len(dor)+len(lat)+len(ven))

# all_sum = dlvsum(allPMNs,dorinds,latinds,veninds)
# e_sum = dlvsum(ePMNs,dorinds,latinds,veninds)
# i_sum = dlvsum(iPMNs,dorinds,latinds,veninds)

# #plot the summed inputs of each PMN to D,L,V groups
# plot_p2dlv_weights(all_sum,gnames,pnames)
# plot_p2dlv_weights(e_sum,gnames,ex_pnames)
# plot_p2dlv_weights(i_sum,gnames,in_pnames)
