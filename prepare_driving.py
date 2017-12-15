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

from config import *


def prepare_historical_run(filename,leapremoved,datastartyear):
    '''
    Data pre-requisite:
    Text file containing any number of columns (one for each variable) and one row per daily data value
    No header is required
    Data must start on 1st January
    
    Input Param: filename: name of the file with the data in it. The data must be daily data and must start on January 1st.
    Input Param: leap: set to 1 if leap years are contained in the data and 0 otherwise
    Input Param: datastartyear: set to the year at the start of the data
    Outputs:
    A tuple containing two arrays: data with leaps removed; data with leaps not removed
    '''    
    data = np.genfromtxt(filename)
    dataorig = data
    if leapremoved == 0:
        if datastartyear % 4 == 1: # if the start year is not a leap year (Matthew)
            for t in range(424,len(data),1459): 
                data = np.delete(data, (t),axis=0)
        elif datastartyear % 4 == 2: # if the start year is not a leap year (Mark)
            for t in range(789,len(data),1459): 
                data = np.delete(data, (t),axis=0)
        elif datastartyear % 4 == 3: # if the start year is not a leap year (Luke)
            for t in range(1154,len(data),1459): 
                data = np.delete(data, (t),axis=0)
        elif datastartyear % 4 == 0: # if the start year is a leap year (Jhon)
            for t in range(59,len(data),1459):
                data = np.delete(data, (t),axis=0) 
        else:
            raise ValueError('There is a problem on the datastartyear value. Please check on the config_file.txt')
    np.savetxt('alldata_noleap.txt',data,delimiter=' ',fmt='%6.2f')
    return data, dataorig

def prepare_ensemble_runs(init_year,init_month,init_day,periodstart_year,periodstart_month,periodstart_day,periodend_year,periodend_month,periodend_day,datastartyear,climstartyear,climendyear,leapinit,leaparray,nonleaparray):
    '''
    
    Input parameters:
    init_year: Year of first date weather is unknown (last day of the present/hindcast equivalent)
    init_month: Month of first date weather is unknown (last day of the present/hindcast equivalent)
    init_day: Day of first date weather is unknown (last day of the present/hindcast equivalent)
    periodstart_year: Year to start the hindcast system. This should include the whole period of interest and any spin up  
    periodstart_month: Month to start the hindcast system. This should include the whole period of interest and any spin up 
    periodstart_day: Day to start the hindcast system. This should include the whole period of interest and any spin up 
    periodend_year: Year that the hindcast system runs until. This should extend beyond the period of interest
    periodend_month: Month that the hindcast system runs until. This should extend beyond the period of interest
    periodend_day: Day that the hindcast system runs until. This should extend beyond the period of interest
    datastartyear: Year for which the data starts
    climstartyear: First year of climatology (for weather generator)
    climendyear: Last year of climatology (for weather generator)
    leapinit: 1 to retain leap years in the initialization step; 0 to not retain leap years
    leaparray: array of input data including leap years [if leapinit is set to zero, this can be a dummy variable]
    nonleaparray: array of input data not including leap years
    Outputs:
    nothing returned from the function. The function will write a zip file containing driving data for each ensemble member
    
    '''
    #Ensure that the data have two dimensions
    if len(np.shape(leaparray)) == 1:
        leaparray = np.reshape(leaparray,(len(leaparray),1))
    if len(np.shape(nonleaparray)) == 1:
        nonleaparray = np.reshape(nonleaparray,(len(nonleaparray),1))
    #print(init_month)
    #Identify line in data array for the forecast initialization    
    if leapinit == 1:
        init_index = (dt.date(init_year,init_month,init_day) - dt.date(datastartyear,1,1)).days
    else:
        init_index = (365*(init_year-datastartyear)) + (dt.date(1973,init_month,init_day) - dt.date(1973,1,1)).days #1973 is chosen as an arbitrary non-leap year
            
    #Identify line in data array for the start of the period
    if leapinit == 1:
        periodstart_index = (dt.date(periodstart_year,periodstart_month,periodstart_day) - dt.date(datastartyear,1,1)).days
    else:
        periodstart_index = (365*(periodstart_year-datastartyear)) + (dt.date(1973,periodstart_month,periodstart_day) - dt.date(1973,1,1)).days #1973 is chosen as an arbitrary non-leap year
    
    #Identify the forecast initialization 
    doy_init = (dt.date(1973,init_month,init_day) - dt.date(1973,1,1)).days
    
    #Calculate the number of days between the forecast initialization and the forecast period end date
    
    number_future_days = (dt.date(periodend_year,periodend_month,periodend_day) - dt.date(init_year,init_month,init_day)).days #Note that this is slightly approximate because it may or may not include a leap day in the calculation. But this should not matter as users will be directed to include a forecast period end well after their period of interest
    
    #Calculate the start and end indices in the non-leap file for each forecast ensemble member
    
    forecaststart_index = np.arange((365*(climstartyear-datastartyear)+doy_init),(climendyear-climstartyear+1)*365,365)
    
    forecastend_index = forecaststart_index+number_future_days
    years = np.arange(climstartyear,climendyear+1)
    filenames = []
    for i in np.arange(0,len(forecaststart_index)):
        filenames.append(str("ensrun_")+str(years[i]))
    
    #print(init_index)
    for i in np.arange(0,len(forecaststart_index)):
        if leapinit == 0:
            dataout = np.vstack((nonleaparray[periodstart_index:init_index,:],nonleaparray[forecaststart_index[i]:forecastend_index[i],:]))
        if leapinit == 1:
            dataout = np.vstack((leaparray[periodstart_index:init_index,:],nonleaparray[forecaststart_index[i]:forecastend_index[i],:]))
        
        np.savetxt(filenames[i],dataout,delimiter=' ',fmt='%6.2f')
    call("zip -qq ensdriving.zip ensrun*", shell=True)
    call("rm ensrun*", shell=True)
        
    
    
    
    
    



   
    
        
    
