from config import *
from config_cumrain import *
from prepare_driving import *
from calc_cumulativerainfall import *

#This is called here just to make a non-leap array. The ensemble is prepared using the prepare_driving_wrapper.py script
outdata = prepare_historical_run(filename,leapremoved,datastartyear)

#ECB note need to run prepare_ensemble()
call("unzip -qq ensdriving.zip",shell=True)


dataout = cumrain_ens(periodstart_month,periodstart_day,intereststart_month,intereststart_day,interestend_month,interestend_day,rainfallcolumn,intereststart_year,interestend_year,ensemble_metric_file)
call("zip -qq ensdriving.zip ensrun*", shell=True)
call("rm ensrun*", shell=True)

#This is to make a cumulative rainfall forecast metric file. 
cumrain_hist(outdata[0],datastartyear,intereststart_month,intereststart_day,interestend_month,interestend_day,rainfallcolumn,intereststart_year,interestend_year,climatological_metric_file)

plotts(dataout)