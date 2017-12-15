import numpy as np
import math
import copy
import datetime as dt
import os
import glob
import os
import string
import sys
import warnings
from shutil import copyfile
from subprocess import call 
import matplotlib.pyplot as plt

from config import *
from config_cumrain import *



def cumrain_ens(periodstart_month,periodstart_day,intereststart_month,intereststart_day,interestend_month,interestend_day,rainfallcolumn,climstartyear,climendyear,ensemble_metric_file):
    """Script to calculate the cumulative rainfall for each ensemble member. 
    
    Prerequisite data:
    It is assumed that the ensemble meteorological driving time series are held in a file called ensdriving.zip, which includes files with the naming convention ensrun_<year>
    
    Input arguments:
    Parameter: periodstart_month - start month for each ensemble member file (i.e. ensrun_<year>)
    Parameter: periodstart_day - start day for each ensemble member file (i.e. ensrun_<day>)
    Parameter: intereststart_month - start month for the cumulation period
    Parameter: intereststart_day - start day for the cumulation period
    Parameter: interestend_month - end month for the cumulation period
    Parameter: interestend_day - end day for the cumulation period
    Parameter: rainfallcolumn - column holding the variable of interest
    Parameter: climstartyear - start year of the period of interest
    Parameter  climendyear - end year of the period of interest
    Parameter: ensemble_metric_file - path to file for writing the output data"""
    #Calculate indices of the start and end for the period of interest
    index_start = (dt.date(1973,intereststart_month,intereststart_day)-dt.date(1973,periodstart_month,periodstart_day)).days
    index_end = (dt.date(1973,interestend_month,interestend_day)-dt.date(1973,periodstart_month,periodstart_day)).days
    
    rainfallcolumn = rainfallcolumn - 1
    
    if index_end < index_start:
        index_end = 365+index_end
    
    years = np.arange(climstartyear,climendyear+1)
    filenames = []
    for i in np.arange(0,len(years)):
        filenames.append(str("ensrun_")+str(years[i]))
    
    
    all = np.zeros(1)
    allts = np.genfromtxt(filenames[0])[:,rainfallcolumn]
    #print index_start, index_end
    
    for i in np.arange(0,len(years)):
        #print i, years[i]
        datain = np.genfromtxt(filenames[i])
        
        ts = datain[:,rainfallcolumn]
        
        dataout = np.sum(datain[index_start:index_end,rainfallcolumn])
        
        all = np.append(all,dataout)
        allts = np.vstack((allts,ts))
        
    allts = allts[1:np.shape(allts)[1]].T    
        
    cumrain_ens = np.vstack((years,all[1:len(all)])).T
    
    
    np.savetxt(ensemble_metric_file,cumrain_ens,delimiter=' ',fmt='%6.2f',header="Year Metric")
    return allts


def cumrain_hist(nonleaparray,datastartyear,intereststart_month,intereststart_day,interestend_month,interestend_day,column,intereststart_year,interestend_year,climatological_metric_file):
    """Script to calculate a time series of cumulative rainfall for each ensemble member. 
   
    Input arguments:
    Parameter: nonleaparray - an array containing daily climatological data with leap years removed
    Parameter: datastartyear - year that the data in nonleaparray start
    Parameter: intereststart_month - start month for the cumulation period
    Parameter: intereststart_day - start day for the cumulation period
    Parameter: interestend_month - end month for the cumulation period
    Parameter: interestend_day - end day for the cumulation period
    Parameter: column - column holding the variable of interest
    Parameter: intereststart_year - first year of the period of interest 
    Parameter: interestend_year - last year of the period of interest
    Parameter: climatological_metric_file - path to file for writing the output data
    Parameter: """
    column = column - 1
     
    noyears = interestend_year - intereststart_year + 1
    
    first_index_start = (dt.date(intereststart_year,intereststart_month,intereststart_day) - dt.date(datastartyear,1,1)).days
    first_index_end = (dt.date(intereststart_year,interestend_month,interestend_day) - dt.date(datastartyear,1,1)).days
    
    
    
    if first_index_end < first_index_start:
        first_index_end = first_index_end + 365
        noyears = noyears - 1
    
    start_indices = np.arange(first_index_start,noyears*365,365)
    end_indices = np.arange(first_index_end,noyears*365,365)
    #print(start_indices)
    
    start_indices = start_indices[0:len(end_indices)]
    
    all = np.zeros(1)
    
    
    
    for i in np.arange(0,int(noyears)):
        if calccumulative == 0:
            dataout = np.mean(nonleaparray[start_indices[i]:end_indices[i],column])
        if calccumulative == 1:
            dataout = np.sum(nonleaparray[start_indices[i]:end_indices[i],column])
            
        all = np.append(all,dataout)
    
    all = all[1:len(all)]
    years = np.arange(datastartyear,datastartyear+noyears)
    output = np.vstack((years,all)).T
    np.savetxt(climatological_metric_file,output,delimiter=' ',fmt='%6.2f',header="Year Metric")

def plotts(datain):
    ymax = np.max(datain)
    ymin = np.min(datain)
    #print(np.shape(datain))
    #print(np.shape(datain)[1])
    outdata = np.zeros(np.shape(datain))
    #print np.shape(outdata)
    for i in np.arange(0,np.shape(datain)[1]):
        
        for j in np.arange(0,np.shape(datain)[0]):    
            outdata[j,i] = np.sum(datain[0:j,i])
        
        
        plt.plot(outdata[:,i],color='black',linewidth=0.5)
        
    plt.plot(np.mean(outdata,axis=1),color='red',linewidth=2)
    start_date = dt.date(periodstart_year,periodstart_month,periodstart_day).isoformat()
    init_date = dt.date(init_year,init_month,init_day).isoformat()
    intereststart_index = (dt.date(periodstart_year,intereststart_month,intereststart_day)-dt.date(periodstart_year,periodstart_month,periodstart_day)).days
    interestend_index = (dt.date(periodstart_year,interestend_month,interestend_day)-dt.date(periodstart_year,periodstart_month,periodstart_day)).days
    if interestend_index < intereststart_index:
        interestend_index = interestend_index+365
    plt.axvline(intereststart_index,color="green",linewidth=2)
    plt.axvline(interestend_index,color="green",linewidth=2)
    plt.xlabel(str("Days after ") +str(start_date))
    plt.ylabel("Cumulative Rainfall (mm)")
    plt.title("Cumulative Rainfall forecast for "+str(init_date))
    plt.savefig("cumulativerainfall"+str(init_date)+str(".png"))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    #plt.show()
    
    
#Unzip the driving file



