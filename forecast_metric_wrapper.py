
from forecast_metric import *
from prepare_driving import *

nonleaparray, leaparray = prepare_historical_run(filename,leapremoved,datastartyear)
forecastmetric(nonleaparray,datastartyear,intereststart_year,intereststart_month,intereststart_day,interestend_year,interestend_month,interestend_day,column,calccumulative)    
