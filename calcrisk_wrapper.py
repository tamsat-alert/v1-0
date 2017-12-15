from config import *
from calcrisk import *
from config_calcrisk import *

pp = risk_prob_plot(climstartyear, climendyear, datastartyear, dataendyear, init_year, init_month,init_day, stat, sta_name, weights, climatological_metric_file, ensemble_metric_file, forecast_metric_file)