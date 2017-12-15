from config import *
from prepare_driving import *

outdata = prepare_historical_run(filename,leapremoved,datastartyear)
output = prepare_ensemble_runs(init_year,init_month,init_day,periodstart_year,periodstart_month,periodstart_day,periodend_year,periodend_month,periodend_day,datastartyear,climstartyear,climendyear,leapinit,outdata[1],outdata[0])
 