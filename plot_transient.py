# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:32:34 2020

@author: gejianwen
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt



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