from __future__ import division
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
from config_forecastmetric import *


def forecastmetric(nonleaparray,datastartyear,intereststart_year,intereststart_month,intereststart_day,interestend_year,interestend_month,interestend_day,column,calccumulative):
    """This is a script to calculate a metric for weighting, based on meteorological inputs
    
    Input parameters:
    nonleaparray: an array with any number of columns (one for each meteorological variable) and one row for each day 
    datastartyear: year that nonleaparray starts. The data must start on 1st January
    intereststart_year: the start year for the period of interest for the weighting metric time series
    intereststart_month: the start month for the period of interest for the weighting metric time series
    intereststart_day: the start day for the period of interest for the weighting metric time series
    interestend_year: the end year for the period of interest for the weighting metric time series
    interestend_month: the end month for the period of interest for the weighting metric time series
    interestend_day: the end day for the period of interest for the weighting metric time series
    column: the column for the variable of interest
    calccumulate: How to derive the metric: 1 for cumulation; 0 for mean
    """
    
    column = column - 1
     
    #Should not assume that this is the climatology. 
    noyears = interestend_year - intereststart_year + 1 
    #noyears=climendyear - climstartyear + 1
    first_index_start = (dt.date(climstartyear,intereststart_month,interestend_day) - dt.date(datastartyear,1,1)).days
    first_index_end = (dt.date(climstartyear,interestend_month,interestend_day) - dt.date(datastartyear,1,1)).days
    
    
    #print(str("first_index_end ")+str(first_index_end))
    #print(str("first_index_start ")+str(first_index_start))
    if first_index_end < first_index_start:
        first_index_end = first_index_end + 365
        noyears = noyears - 1
    
    #print(noyears)
    
    start_indices = np.zeros(noyears)
    end_indices = np.zeros(noyears)
    
    start_indices[0] = first_index_start
    end_indices[0] = first_index_end
    for i in np.arange(1,noyears):
        start_indices[i] = start_indices[i-1]+365
        end_indices[i] = end_indices[i-1]+365
    
    
    
    all = np.zeros(1)
    
    
    
    for i in np.arange(0,int(noyears)):
        if calccumulative == 0:
            dataout = np.mean(nonleaparray[int(start_indices[i]):int(end_indices[i]),int(column)])
        if calccumulative == 1:
            dataout = np.sum(nonleaparray[int(start_indices[i]):int(end_indices[i]),int(column)])
            
        all = np.append(all,dataout)
    
    all = all[1:len(all)]
    years = np.arange(datastartyear,datastartyear+noyears)
    output = np.vstack((years,all)).T
    np.savetxt(forecast_metric_file,output,delimiter=' ',fmt='%6.2f',header="Year Metric")
    


    