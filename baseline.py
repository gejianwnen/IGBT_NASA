#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


import pandas as pd
import numpy as np
# import sklearn

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
from sklearn.preprocessing import MinMaxScaler


# # slice data while switching 
# ### 1. up-edge 
# ### 2. down-edge

# In[31]:


# define plot and save function 
def plot_transient(sample_df,sample_name,save_dir):
    # plot the time sequence parameter
    plt.figure(figsize=(16, 12))
    plt.subplot(221)
    plt.plot(sample_df.index.values, sample_df.gateSignalVoltage.values)
    plt.title('Gate Signal Voltage',fontsize ='xx-large' )
    plt.xlabel('Time /10ns',fontsize ='xx-large')
    plt.ylabel('Voltage /V',fontsize ='xx-large')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-2,12])
    plt.tight_layout()
    # plt.savefig( 'transient_picture/gateSignalVoltage'+sample_name+'.png')

    plt.subplot(222)
    plt.plot(sample_df.index.values, sample_df.gateEmitterVoltage.values)
    plt.title('Gate Emitter Voltage',fontsize ='xx-large' )
    plt.xlabel('Time /10ns',fontsize ='xx-large')
    plt.ylabel('Voltage /V',fontsize ='xx-large')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-2,12])
    plt.tight_layout()
    # plt.savefig( 'transient_picture/gateEmitterVoltage'+sample_name+'.png')

    # plot the time sequence parameter
    plt.subplot(223)
    plt.plot(sample_df.index.values, sample_df.collectorEmitterVoltage.values)
    plt.title('Collector Emitter Voltage',fontsize ='xx-large' )
    plt.xlabel('Time /10ns',fontsize ='xx-large')
    plt.ylabel('Voltage /V',fontsize ='xx-large')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-2,12])
    plt.tight_layout()
    # plt.savefig('transient_picture//collectorEmitterVoltage'+sample_name+'.png')

    # collectorEmitterCurrentSingal
    # plot the time sequence parameter
    plt.subplot(224)
    plt.plot(sample_df.index.values, sample_df.collectorEmitterCurrentSingal.values)
    plt.title('Collector Emitter Current',fontsize ='xx-large' )
    plt.xlabel('Time /10ns',fontsize ='xx-large')
    plt.ylabel('Current /A',fontsize ='xx-large')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-2,12])
    plt.tight_layout()
    plt.savefig(save_dir+sample_name+'.png')

    plt.clear("all")
    


# In[32]:


# slice data
# define local directions
file_forder = "./transient_timeDomain/" 
up_dir = "./up_edge/"
down_dir = "./down_edge/"
up_pic_dir = "./up_edge_pic/"
down_pic_dir = "./down_edge_pic/"

for (_ ,_ , process_files) in os.walk(file_forder):
    for file in process_files:
        sample_df = pd.read_csv(file_forder + file )
        sample_name = file[:-4]
        # get index in up and down edge
        n = np.where(sample_df.gateSignalVoltage.values>8)  # return a tuple so use n[0][0] to index
        # slice data
        up_df = sample_df.iloc[n[0][0]-100:n[0][0]+924,:]
        down_df = sample_df.iloc[n[0][-1]-100:n[0][-1]+924,:]
        # save data
        up_df.to_csv(up_dir+file,index = False)
        down_df.to_csv(down_dir+file,index = False)
        # plot data
#         plot_transient(down_df,sample_name, down_pic_dir)
#         plot_transient(up_df,sample_name,up_pic_dir)

# show info
# print(sample_df.shape)
# print(sample_df.columns)
# sample_df.head()


# # featuring
# ### data process have done previousely
# ### every transient time damain data was sliced

# In[2]:


def cal_feature(df, col):
    x = df[col].values
    x_min = df[col].min()
    x_max = df[col].max()
    x_range = x_max-x_min
    x_mean = df[col].mean()
    x_mode = df[col].mode()[0]
    x_median = df[col].median()
    x_quantile25 = df[col].quantile( q=0.25)
    x_quantile75 = df[col].quantile( q=0.75)
    x_std = df[col].std()
    x_var = df[col].var()
    x_skew = df[col].skew()
    x_sem = df[col].sem()                   # standard error of the mean
    x_mad = df[col].mad()                   # the mean absolute deviation of the values
    x_kurt = df[col].kurt()                 # 偏度是三阶中心矩和标准差的三次方的比值 峭度是四阶中心矩和标准差的四次方的比值
    x_msv = np.mean(df[col].pow(2))         # 均方值(mean-square value) 代表了信号的能量 msv
    x_rms = np.sqrt(x_msv)                  # 均方根，均方值的根 root of mean square valve  rmsv
    x_arv = np.mean(df[col].abs())          # 绝对均值(整流平均值arv-Average rectified value，或者叫mean of absolute value) mav
    x_sra = df[col].abs().pow(0.5).mean()**2  # (square root amplitude)   smrv 方根幅值
    x_I = x_max/x_arv                         # 脉冲因子
    x_sf = x_rms/x_arv                        # 波形因子(Form factor&shape factor) 有效值（RMS）与整流平均值的比值
    x_MI = x_range/x_sra                      # 裕度因子 信号峰值与方根幅值的比值
    x_CF = x_range/x_rms                      # 峰值因子(Crest factor),代表峰值在波形中的极端程度。
    x_llr = np.mean(np.log(np.abs(x)+1))/np.log(x_std)     # log-log ratio
    x_pi = x_max/x_mean                       # pulse indicators
    x_sdif = x_std/x_arv                      # SDIF
    x_cpt1 = max(abs(x))/x_sra                #CPT1
    feature_cols = ['x_min', 'x_max', 'x_range', 'x_mean', 'x_mode' ,'x_median' ,'x_quantile25', 'x_quantile75', 'x_std', 'x_var',
                    'x_skew', 'x_sem', 'x_mad', 'x_kurt',  'x_msv', 'x_rms',  'x_arv' ,  'x_sra',   'x_I',    'x_sf',
                    'x_MI',  'x_CF',   'x_llr', 'x_pi',   'x_sdif', 'x_cpt1']
    feature_cols = [s.replace("x_",col+"_") for s in feature_cols]
    feature_list = [x_min, x_max, x_range, x_mean, x_mode ,x_median ,x_quantile25, x_quantile75, x_std, x_var,
                    x_skew, x_sem, x_mad, x_kurt,  x_msv,  x_rms,   x_arv ,   x_sra,    x_I,   x_sf, 
                    x_MI,   x_CF,  x_llr,  x_pi,  x_sdif,  x_cpt1]
    feature_df = pd.DataFrame(data = feature_list).T
    feature_df.columns = feature_cols
    
    return feature_df


# In[3]:


# define local directions
up_dir = "./up_edge/"
down_dir = "./down_edge/"

# down edge
all_feature_df = pd.DataFrame()
for (_ ,_ , process_files) in os.walk(down_dir):
    for file in process_files:
        # read file
        sample_df = pd.read_csv(down_dir+file)
        # drop some cols and expand feature P
        sample_df = sample_df.drop(["collectorEmitterCurrentSingal","gateSignalVoltage"],axis = 1)
        sample_df["P"] = sample_df["gateEmitterVoltage"]*sample_df["collectorEmitterVoltage"]
        # sample_df["P_diff"] = sample_df["P"].diff()
        # sample_df["gateEmitterVoltage_diff"] = sample_df["gateEmitterVoltage"].diff()
        sample_df.fillna(method = "backfill",inplace = True)
        # calculate features
        cols = sample_df.columns
        feature_df = pd.DataFrame()
        for col in cols:
            df = cal_feature(sample_df, col)
            feature_df = pd.concat([feature_df,df],axis = 1)
        feature_df["Vce_pre"] = np.mean(sample_df['collectorEmitterVoltage'].values[:60])
        feature_df["P_pre"] = np.mean(sample_df['P'].values[:60])
        feature_df['round'] = int(file[0:-4])
        
        all_feature_df = pd.concat([all_feature_df,feature_df],axis = 0)

all_feature_df.sort_values(by = 'round', inplace=True)    
all_feature_df.to_csv('./temp/down_edge_features.csv',index = False)

# up edge
all_feature_df = pd.DataFrame()
for (_ ,_ , process_files) in os.walk(down_dir):
    for file in process_files:
        # read file
        sample_df = pd.read_csv(up_dir+file)
        # drop some cols and expand feature P
        sample_df = sample_df.drop(["collectorEmitterCurrentSingal","gateSignalVoltage"],axis = 1)
        sample_df["P"] = sample_df["gateEmitterVoltage"]*sample_df["collectorEmitterVoltage"]
        # sample_df["P_diff"] = sample_df["P"].diff()
        # sample_df["gateEmitterVoltage_diff"] = sample_df["gateEmitterVoltage"].diff()
        sample_df.fillna(method = "backfill",inplace = True)
        # calculate features
        cols = sample_df.columns
        feature_df = pd.DataFrame()
        for col in cols:
            df = cal_feature(sample_df, col)
            feature_df = pd.concat([feature_df,df],axis = 1)
        feature_df["Vce_post"] = np.mean(sample_df['collectorEmitterVoltage'].values[-60:])
        feature_df["P_post"] = np.mean(sample_df['P'].values[-60:])
        feature_df['round'] = int(file[0:-4])
        
        all_feature_df = pd.concat([all_feature_df,feature_df],axis = 0)

all_feature_df.sort_values(by = 'round', inplace=True)    
all_feature_df.to_csv('./temp/up_edge_features.csv',index = False)

print(all_feature_df.shape)
all_feature_df.head()


# In[5]:


# conband the up and down features
up_df = pd.read_csv('./temp/up_edge_features.csv')
down_df = pd.read_csv('./temp/down_edge_features.csv')
up_cols = ['up_'+c for c in up_df.columns]
down_cols = ['down_'+c for c in down_df.columns]
up_df.columns = up_cols
down_df.columns = down_cols
up_df['round'] = up_df['up_round']
up_df.drop(['up_round'],axis = 1,inplace = True)
down_df['round'] = down_df['down_round']
down_df.drop(['down_round'],axis = 1,inplace = True)
train_df = pd.merge(up_df,down_df, how="left",on = "round" )

train_df.to_csv("./temp/train_feature_df.csv",index = False)
print(train_df.shape)
train_df.head()


# In[10]:


# add temperature
train_df = pd.read_csv('./temp/train_feature_df.csv')
tempareture_df = pd.read_csv("collectorEmitterVoltage_peak_temperature.csv")
train_df["temperature"] = tempareture_df["tempareture"]
# change the target num
train_df["num_cycle"] = train_df["round"]
train_df.drop(['round'],axis = 1, inplace = True)

train_df.to_csv("./temp/train_feature_df.csv",index = False)

print(train_df.shape)
train_df.head()


# In[19]:


def plot_feature1(df, base_col, save_dir):
    plt.figure(figsize=(12, 6))
    cols = [c for c in  df.columns if c!=base_col ]
    for col in cols:
        plt.plot(df[base_col].values ,df[col].values)
        plt.title(col,fontsize = 'x-large')
        plt.tight_layout()
        plt.savefig(save_dir+col+'.png')
        plt.clf()

# drop outl
round_num_del = [1,2,3,4,5,6,7,8,9,10,11,12,13,24,35,46,112,223,334,386,375,364,
                 56,153,208,309,327,347,351,360,365,391]
round_num_del = [i-1 for i in round_num_del]
up_pic_dir = "./up_edge_pic/"
down_pic_dir = "./down_edge_pic/"

train_df = pd.read_csv('./temp/down_edge_features.csv')
train_df.drop(round_num_del,axis = 0,inplace = True)
train_df = train_df.loc[train_df['temperature']>326,:]
plot_feature1(train_df,"round",down_pic_dir)

train_df = pd.read_csv('./temp/up_edge_features.csv')
train_df.drop(round_num_del,axis = 0,inplace = True)
plot_feature1(train_df,"round",up_pic_dir)

# show info
print(train_df.shape)
train_df.head()


# ## filter

# In[21]:


def myMedFilter(s):
    s[2:-2] = np.median([s[0:-4],s[1:-3],s[2:-2],s[3:-1],s[4:]],axis = 0)
    s[1] = np.median(s[:3])    
    s[-2] = np.median(s[-3:])
    
    return s
    
    
def plot_feature2(df, base_col, save_dir):
    plt.figure(figsize=(12, 6))
    cols = [c for c in  df.columns if c!=base_col ]
    for col in cols:
        plt.plot(df[base_col].values ,df[col].values)
        plt.title(col,fontsize = 'x-large')
        plt.tight_layout()
        plt.savefig(save_dir+'filtered_'+col+'.png')
        plt.clf()

# drop outl
round_num_del = [1,2,3,4,5,6,7,8,9,10,11,12,13,24,35,46,112,223,334,386,375,364,
                 56,153,208,309,327,347,351,360,365,391]
round_num_del = [i-1 for i in round_num_del]
up_pic_dir = "./up_edge_pic/"
down_pic_dir = "./down_edge_pic/"


train_df = pd.read_csv('./temp/down_edge_features.csv')
train_df.drop(round_num_del,axis = 0,inplace = True)
cols = [c for c in train_df.columns if c!="round"]
for c in cols:
    train_df[c] = myMedFilter(train_df[c].values)
plot_feature2(train_df,"round",down_pic_dir)

train_df = pd.read_csv('./temp/up_edge_features.csv')
train_df.drop(round_num_del,axis = 0,inplace = True)
cols = [c for c in train_df.columns if c!="round"]
for c in cols:
    train_df[c] = myMedFilter(train_df[c].values)
plot_feature2(train_df,"round",up_pic_dir)

# show info
print(train_df.shape)
train_df.head()


# # correlation

# In[11]:


train_df = train_df.loc[train_df['temperature']>326,:]
train_df['temperature'].hist()


# In[43]:


train_df = pd.read_csv('./temp/train_feature_df.csv')
corr_df = train_df.corr().abs()
# corr_df.to_csv("temp/coelationship_outliners.csv")
# plot
im =np.abs(corr_df.values) 
plt.figure(figsize = (10,10))
plt.imshow(im)
plt.tight_layout()
plt.savefig('temp/correlationship'+'.png')


# # feature select

# In[3]:


cols =     ['down_gateEmitterVoltage_MI',
#             'down_gateEmitterVoltage_mad',
            'down_collectorEmitterVoltage_skew',
#             'down_gateEmitterVoltage_sf',
#             'down_gateEmitterVoltage_sra',
            'down_gateEmitterVoltage_kurt',
            'down_gateEmitterVoltage_llr',
#             'down_collectorEmitterVoltage_llr',
            'up_collectorEmitterVoltage_std',
#             'up_collectorEmitterVoltage_sem',
#             'up_collectorEmitterVoltage_var',
            'up_collectorEmitterVoltage_llr',
#             'up_collectorEmitterVoltage_mad',       # up to 0.6
            "temperature",
            "num_cycle"
           ]
 
train_df = pd.read_csv('/temp/train_feature_df.csv')
feature_df = train_df.loc[:,cols]
feature_df.to_csv('./temp/useful_feature_df.csv',index = False)
corr_df = feature_df.corr().abs()
corr_df.to_csv("temp/correlationship_usefue_features.csv")
# plot
im =np.abs(corr_df.values) 
im = im.round(2)
myHeatMap(im,cols,cols)


# In[2]:


def myHeatMap(harvest,col1,col2):
    farmers = col1
    vegetables = col2
    fig,ax = plt.subplots(figsize = (9,9))
    im = ax.imshow(harvest)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig('temp/correlationship_usefue_features'+'.png')
    plt.show()
    


# In[4]:



# std
feature_df = pd.read_csv('./temp/useful_feature_df.csv')
target = ["temperature","num_cycle"]
cols_to_use = [c for c in cols if c not in target]
feature_df["temperature_scale"] = feature_df["temperature"]
fliter = MinMaxScaler()
for col in cols_to_use:
    feature_df[col] = fliter.fit_transform(feature_df[col].values.reshape(-1, 1))
    
X = feature_df[cols_to_use].values
feature_df.to_csv( "temp/train_df_scale.csv")
feature_df.head()


# In[7]:


# PCA
# dimension reduce
pca = decomposition.PCA(n_components = 2)
X_reduce = pca.fit_transform(X)

# save map
map_df = pd.DataFrame()
map_df['num_cycle'] = feature_df['num_cycle']
map_df['temperature'] = feature_df['temperature_scale']
map_df['x'] = 0
map_df['y'] = 0
map_df[['x','y']] = X_reduce
map_df.to_csv('output/PCA'+'_map.csv',index = False)

def myScatter(map_df,suffix_str):
    # plot it
    cm=plt.cm.inferno  # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    x = map_df['x'].values
    y = map_df['y'].values
    color = map_df['num_cycle'].values
    plt.figure(figsize = (16,7))
    plt.subplot(121)
    sc = plt.scatter(x, y, c=color, vmin=0, vmax=410, s=35, cmap=cm)
    plt.colorbar(sc)

    plt.subplot(122)
    color = map_df['temperature_scale'].values
    sc = plt.scatter(x, y, c=color, vmin=0, vmax=1, s=35, cmap=cm)
    plt.colorbar(sc)

    plt.tight_layout()
    plt.savefig("output/"+"map_"+suffix_str+".png")
    
myScatter(map_df)    


# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




