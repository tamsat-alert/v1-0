#---------------------------------------------------------------------#
# This script is a more generalized TAMSAT-ALERT risk assessment tool.
# Dagmawi Teklu Asfaw
# November, 2017
#---------------------------------------------------------------------#
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy import signal
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns
import warnings
from config_calcrisk import *
import matplotlib.mlab as mlab
# -------------------------------------------------------------------#
# calculate the risk probability and present results
#--------------------------------------------------------------------#
def risk_prob_plot(climastartyear, climaendyear, datastartyear, dataendyear, forecastyear, forecastmonth,
                   forecastday, stat, sta_name, weights, 
                    climafile, forecastfile, weightfile):
    """
    This function plot the probability estimates for poor yield for a single date
    forecast given in the configuration file.

    :param climastartyear: the year climatology value start.
    :param climaendyear: the year climatology value end. 
    :param datastartyear: the year the data set start
    :param dataendyear: the the year the data set end
    :param forecastyear: the year for which we are going to forecast yield
                         from historical climatic weather.
    :param forecastmonth: the month for which we are going to forecast yield
                         from historical climatic weather.
    :param forecastday: the day forecast start.(The last day for which
                        the forecast year has a data)
    :param stat: statistical method to be used for probability distribution comparison (ecdf or norm)
    :param sta_name: name of station or location
    :param wth_path: the file path where the .wth files are present (as string)
    :param weights: tercile forecast probabilities of the weighting metric used
    :param climafile: file containg a climatology (i.e. historical time series) of the metric under investigation. The text file has two columns year value (with a one line header)
    :param forecastfile: file containg ensembles forecast values of the metric under investigation (i.e. each value is the ensemble member associated with the weather future for the year in column 1). The text file has two columns year value (with a one line header)
    :param weightfile: file containg the values of the metric being used for weighting. The text file has two columns. Column 1: Year, Column 2: Value of the metric we want to weight by.
  
    """
  
    #------------------------------------------------------------------#
    # creating folders to put output data and plot
    #------------------------------------------------------------------#
    if not os.path.isdir("./plot_output"):
        os.makedirs("./plot_output")
    if not os.path.isdir("./plot_output/gaussian"):
        os.makedirs("./plot_output/gaussian")
    if not os.path.isdir("./plot_output/ecdf"):
        os.makedirs("./plot_output/ecdf")
    if not os.path.isdir("./data_output"):
        os.makedirs("./data_output")
   
    # set up actual dates for the x axis representation
    date = dt.datetime(forecastyear, forecastmonth, forecastday).date()
    f_date = date.strftime('%d-%b-%Y')

   
    climayears = np.arange(climastartyear, climaendyear+1)

    # read climatology time series 
    climametric = np.genfromtxt(climafile,skip_header=1)[:,1]

    # read forecast ensemble time series 
    forecametric = np.genfromtxt(forecastfile,skip_header=1)[:,1]

    # read Weighting metric time series
    Wmetric = np.genfromtxt(weightfile,skip_header=1)[:,1]

    #----------------------------------------------------------------#
    # calculating probability distribution
    #----------------------------------------------------------------#
    # threshold probability 
    thresholds = np.arange(0.01,1.01,0.01)

    # calculate the mean and sd of the climatology
    climamean = np.mean(climametric)
    climasd = np.std(climametric)
    
    
    # calcualte the mean and sd of the the projected
    # yield based on climatology weather data
    # we need the weighted yield frorecast

    
    fdate = f_date
    yr = forecastyear
    #print(forecastfile,weightfile)
    projmean, projsd = weight_forecast(forecastfile, weightfile, weights,climastartyear,climaendyear)
    projsd = np.maximum(projsd,0.001)  # avoid division by zero
    
    if stat == 'normal':
        # calculate the normal distribution
        probabilityyields = []
        probabilityclim = []
        for z in range(0,len(thresholds)):
            thres = sps.norm.ppf(thresholds[z],climamean,climasd)
            #print(thres)
            #(climamean,climasd)
            #print(projmean,projsd)
            #probyield = sps.norm.cdf(thres,climamean,climasd)
            #print(probyield)
            probclim = sps.norm.cdf(thres,climamean,climasd)
            probyield = sps.norm.cdf(thres,projmean,projsd)
            #print(probyield)
            probabilityyields = np.append(probabilityyields,probyield)
            probabilityclim = np.append(probabilityclim,probclim)
            
            del probyield 
         
        out = np.vstack((probabilityclim,probabilityyields))       
        np.savetxt('./data_output/probyield_normal.txt',out.T,fmt='%0.2f')
       
    elif stat == 'ecdf':
        # calculate the emperical distribution
        ecdf_clima = ECDF(climametric)
        probabilityyields = []
        probabilityclim = []
        
        for z in range(0,len(ecdf_clima.x)):
            thres = ecdf_clima.x[z] #(thresholds[z])
            ecdf_proj = ECDF(forecametric)
            probclim = ecdf_clima(thres)
            probyield = ecdf_proj(thres)
            probabilityyields = np.append(probabilityyields,probyield)
            probabilityclim = np.append(probabilityclim,probclim)
            
            del probyield
           
        
        out = np.vstack((probabilityclim,probabilityyields))
        

        np.savetxt('./data_output/probyield_ecdf.txt',out.T,fmt='%0.2f')
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')

    #-------------------------------------------------------------------#
    # Plots of results
    #-------------------------------------------------------------------#
    # Risk probability plot (origional format ECB)
    sns.set_style("ticks")
    fig = plt.figure()
    ax = plt.subplot(111)
    if stat == 'normal':
        # Plot using normal distribution
        plt.plot(thresholds*100,thresholds,'--k',lw=1, label = 'Climatology') 
        line = plt.plot(thresholds*100,probabilityyields,'k',lw=1, label='Projected')
        # indicating critical points
        highlight_point(ax,line[0],[thresholds[79]*100,probabilityyields[79]],'g') # below average
        highlight_point(ax,line[0],[thresholds[59]*100,probabilityyields[59]],'y') # below average
        highlight_point(ax,line[0],[thresholds[39]*100,probabilityyields[39]],'m') # below average
        highlight_point(ax,line[0],[thresholds[19]*100,probabilityyields[19]],'r') # well below average
      
    elif stat == 'ecdf':
        # Plot using emperical cumulative distribution
        
        plt.plot(ecdf_clima.y*100,ecdf_clima.y,'--k',lw=1, label = 'Climatology') 
        line = plt.plot(ecdf_clima.y*100,probabilityyields,'k',lw=1, label='Projected')
        # identifying the index for the critical points
        nn = int(round(len(climayears)/5.,0)) # this should be an intiger
        wba_i = nn 
        ba_i = (nn * 2) 
        a_i = (nn * 3) 
        av_i = (nn * 4) 
        # indicating critical points
        highlight_point(ax,line[0],[ecdf_clima.y[av_i]*100,probabilityyields[av_i]],'g') # below average
        highlight_point(ax,line[0],[ecdf_clima.y[a_i]*100,probabilityyields[a_i]],'y') # below average
        highlight_point(ax,line[0],[ecdf_clima.y[ba_i]*100,probabilityyields[ba_i]],'m') # below average
        highlight_point(ax,line[0],[ecdf_clima.y[wba_i]*100,probabilityyields[wba_i]],'r') # well below average
    
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')

    plt.title('Theme: Probability of metric estimate (against ' +str(climastartyear)+'-'+ str(climaendyear)+' climatology)\nLocation: ' +sta_name+'\nForecast date: '+ f_date,loc='left',fontsize=14)
    plt.xlabel('Climatological percentile',fontsize=14)
    plt.ylabel('Probability <= Climatological percentile',fontsize=14)
  
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    if stat == 'normal': 
        path = './plot_output/gaussian/'
    elif stat == 'ecdf':
        path = './plot_output/ecdf/'
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')
    plt.savefig(path + sta_name+'_'+f_date+'_yieldprob.png',dpi=300)
    plt.close()

    #-------------------------------------------------------------------------#
    # Risk probability plot (Pentile bar plot format DA)
    pp = []
    sns.set_style("ticks")
    fig = plt.figure()
    if stat == 'normal':
        verylow = probabilityyields[19] 
        low = probabilityyields[39] - verylow
        average = probabilityyields[59] - (verylow+low)    
        high = probabilityyields[79] - (verylow+low+average)
        veryhigh = 1 - (verylow+low+average+high)
    elif stat == 'ecdf':
        # identifying the index for the critical points
        nn = int(round(len(climayears)/5.,0)) # this should be an intiger
        wba_i = nn 
        ba_i = (nn * 2) 
        a_i = (nn * 3) 
        av_i = (nn * 4) 
        
        verylow = probabilityyields[wba_i] 
        low = probabilityyields[ba_i] - probabilityyields[wba_i] #verylow
        average = probabilityyields[a_i] - probabilityyields[ba_i]#(verylow+low)    
        high = probabilityyields[av_i] - probabilityyields[a_i]#(verylow+low+average)
        veryhigh = 1 - probabilityyields[av_i]#(verylow+low+average+high)
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')    

    val = [verylow,low,average,high,veryhigh]   # the bar lengths
    pos = np.arange(5)+.5        # the bar centers on the y axis
    plt.barh(pos[0],val[0]*100, align='center', color='r',label='Very low (0-20%)')
    plt.barh(pos[1],val[1]*100, align='center', color='m',label='Low (20-40%)')
    plt.barh(pos[2],val[2]*100, align='center', color='grey',label='Average (40-60%)')
    plt.barh(pos[3],val[3]*100, align='center', color='b',label='High (60-80%)')
    plt.barh(pos[4],val[4]*100, align='center', color='g',label='Very high (80-100%)')
    
    plt.annotate(str(round(val[0]*100,1))+'%',((val[0]*100)+1, pos[0]),xytext=(0, 1), textcoords='offset points',fontsize=20)
    plt.annotate(str(round(val[1]*100,1))+'%',((val[1]*100)+1, pos[1]),xytext=(0, 1), textcoords='offset points',fontsize=20)
    plt.annotate(str(round(val[2]*100,1))+'%',((val[2]*100)+1, pos[2]),xytext=(0, 1), textcoords='offset points',fontsize=20)
    plt.annotate(str(round(val[3]*100,1))+'%',((val[3]*100)+1, pos[3]),xytext=(0, 1), textcoords='offset points',fontsize=20)
    plt.annotate(str(round(val[4]*100,1))+'%',((val[4]*100)+1, pos[4]),xytext=(0, 1), textcoords='offset points',fontsize=20)

    plt.yticks(pos, ('Very low', 'Low', 'Average','High','Very high'),fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Probability percentile',fontsize=14)
    plt.title('Theme: Probability of metric estimate (against ' +str(climastartyear)+'-'+ str(climaendyear)+' climatology)\nLocation: ' +sta_name+'\nForecast date: '+ f_date,loc='left',fontsize=14)
    plt.xlim(0,101)
    plt.legend()
    plt.tight_layout()
    if stat == 'normal': 
        path = './plot_output/gaussian/'
    elif stat == 'ecdf':
        path = './plot_output/ecdf/'
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')

    # append the probabilities to pp
    pp = np.append(pp,round(val[0]*100,1))
    pp = np.append(pp,round(val[1]*100,1))
    pp = np.append(pp,round(val[2]*100,1))
    pp = np.append(pp,round(val[3]*100,1))
    pp = np.append(pp,round(val[4]*100,1))
    
    plt.savefig(path + sta_name+'_'+f_date+'_pentile.png', dpi=300)
    plt.close()
    
    # save the probabilities of each category on a text file
    headval = '1 = Very low(0-20%)  2 = Low(20-40%)   3 = Average(40-60%)  4 = High(60-80%)  5 = Very high(80-100%)\n\
Category    RiskProbability'
    category = [1,2,3,4,5]
    rp = np.array([category, pp])
    rp = rp.T
    np.savetxt('./data_output/RiskProbability.txt', rp, delimiter = '     ', header=headval, fmt='%i  %0.2f')

    #-------------------------------------------------------------------------#
    # probability density plot
    sns.set_style("ticks")
    fig = plt.figure()

    if stat == 'normal':
        # Plot using normal distribution
        sns.kdeplot(climametric, bw=10, shade=True,label='Climatology',cumulative=False)
        sns.kdeplot(forecametric , bw=10, shade=False,color='g',label='Projected',cumulative=False)

    elif stat == 'ecdf':
        # Plot using emperical cumulative distribution
        sns.kdeplot(climametric, bw=10, shade=True,label='Climatology',cumulative=False)
        sns.kdeplot(forecametric , bw=10, shade=False,label='Projected',cumulative=False)

    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')
    plt.title('Theme: Probability of metric estimate (against ' +str(climastartyear)+'-'+ str(climaendyear)+' climatology)\nLocation: ' +sta_name+'\nForecast date: '+ f_date,loc='left',fontsize=14)
    plt.xlabel('Yield (Kg/ha)',fontsize=14)
    plt.ylabel('Probability density',fontsize=14)
  
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    if stat == 'normal': 
        path = './plot_output/gaussian/'
    elif stat == 'ecdf':
        path = './plot_output/ecdf/'
    else:
        raise ValueError('Please use only "normal" or "ecdf" stat method')
    plt.savefig(path +sta_name+'_'+f_date+'_ked_plot.png', dpi=300)
    plt.close()


    fig2 = plt.figure()
    
    
    
    alldata = np.append(climametric,forecametric)
    #binBoundaries = np.linspace(min(forecametric), max(forecametric),10)
    
    
    
    binBoundaries = np.linspace(min(alldata), max(alldata),10)
    
    thres = np.zeros(len(binBoundaries))
    probclim = np.zeros(len(binBoundaries))
    modfreqclim = np.zeros(len(binBoundaries))
    probclim[0] = sps.norm.cdf(binBoundaries[0],climamean,climasd)
    for z in range(1,len(binBoundaries)):
        thres[z] = binBoundaries[z]
        
        probclim[z] = sps.norm.cdf(thres[z],climamean,climasd)
        modfreqclim[z] = (probclim[z] - probclim[z-1])
        
        #probyield = sps.norm.cdf(thres,projmean,projsd)
        #probabilityyields = np.append(probabilityyields,probyield)
        #probabilityclim = np.append(probabilityclim,probclim)
    
    #plt.hist(forecametric, bins=binBoundaries, color = 'b',lw=3)
    n, bins, patches = plt.hist((climametric,forecametric),bins=binBoundaries, lw=3, color = ["blue","green"],label=["Climatology","Ensemble"],normed=True)
    
    #plt.plot(binBoundaries,modfreqclim)
    #https://plot.ly/matplotlib/histograms/
    y = mlab.normpdf(bins, climamean, climasd)
    plt.plot(bins,y,"b",label="Climatological modelled distribution")

    #forecamean = np.mean(forecametric)
    #forecasigma = np.std(forecametric)
    y2 = mlab.normpdf(bins,np.mean(forecametric),np.std(forecametric))

    plt.plot(bins,y2,"g",label="Ensemble modelled distribution")

    y3 = mlab.normpdf(bins,projmean,projsd)
    plt.plot(bins,y3,"g",ls="--",label="Weighted ensemble modelled distribution")
    plt.xlabel('Metric value',fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.title('Histogram of ensemble predictions (compared against ' +str(climastartyear)+'-'+ str(climaendyear)+' climatology)\nLocation: ' +sta_name+'\nForecast date: '+ f_date,loc='left',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(min(alldata), max(alldata))
    #plt.ylim(0,len(forecametric)+1)
    plt.legend()
    plt.tight_layout()
                 
    plt.savefig(path +sta_name+'_'+f_date+'_hist_plot.png', dpi=300)
    plt.close()
    

    #plot additional variables of the input data
    #cum_plots(climastartyear, climaendyear, forecastyear, sta_name, wth_path, weights)

    return pp

#--------------------------------------------------------------------------------#
        
def highlight_point(ax,line,point, c, linestyle=':'):
    """
    This is an extra function to highlight three of the probability
    points on the plot. It is part of the main plotting function.
    """
    label = ['well below average = ', 'Below average = ', 'Average = ','Above average = ']
    c = c 
    xmin = 0 #ax.get_xlim()[0]
    ymin = 0 #ax.get_ylim()[0]
    if c == 'r':
        label = label[0]
    elif c == 'm':
        label = label[1]
    elif c == 'y':
        label = label[2]
    elif c == 'g':
        label = label[3]
    else:
        raise ValueError('Only chosse colors green,yellow or red')
    ax.plot([xmin,point[0]],[point[1],point[1]],color=c,linestyle=linestyle, label = label+str(round(point[1],2)))
    ax.plot([point[0],point[0]],[ymin,point[1]],color=c,linestyle=linestyle)
    return None

#-------------------------------------------------------------------------------#  
def weight_forecast(forecastfile, weightfile, weights,climastartyear,climaendyear):
    #Need to implement weights for ecdf. https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    
    fy_wmean = []
    

    # read forecast ensemble time series (This file is created during crop yield forecast)
    forecametric = np.genfromtxt(forecastfile,skip_header=1)[:,1]

    # read Weighting metric time series (This file is created by weighting metric prep)
    # this only work for GLAM format so one has to write its own code or provide specific
    # file with two column 1, climayears 2, weight metric value (header must be given in the file)
    Wmetric = np.genfromtxt(weightfile,skip_header=1)[:,1]
    
    
    # the metric for ordering the true metric(forecametric)
    # is total precipitation of JJA.

    #ECB: Note that if we have fewer years for the weighting metric, which can happen if we go over the year boundary, zip will cut out the end of the ensembles time series. This should be sorted out now. The onus is on the user to specify the correct years.
    #ECB: Note that when we cross the year boundary in either ensemble period of interest or weighting metric period of interest, we reference the FIRST year in the period. So if we are using a DJF weighting metric period of interest, for a MAM ensemble period of interest, we will be weighting a using a metric AFTER the period of interest.  This needs to be resolved. The user needs to specify whether our forecast period starts in the same year, the year before or the year after.
    climayears = np.arange(climastartyear, climaendyear+1)

    #----------------------------------------------------------------#
    # warning that certain number of years have been removed from the
    # climatology to make the length divisioble by len(weight)
    #-----------------------------------------------------------------#
    ny_del = len(climayears) % len(weights)
    #print(len(climayears))
    if ny_del != 0:
         climayears = climayears[:(len(climayears) - ny_del)]
         #warnings.warn("The last%s ,"year of climatology years has been removed!" % ny_del)
         #print "Only", %s, "ensembles are used!" % len(climayears)
    else:
        climayears = climayears
    
        
    forecametric = forecametric[0:len(climayears)]
    Wmetric = Wmetric[0:len(climayears)]
    
        
    out = zip(Wmetric,forecametric) # put the metric with true metric (forecametric)
   
    out = sorted(out) # sort in  ascending order based on the metric
    out = np.array(out) # convert it to array
    #print(out)
    # weighting forecast metric with the weighting metric
    n_reps = np.shape(out)[0] / len(weights) 
    allweights = []
    
    for j in range(0,len(weights)):
        allweights = np.append(allweights,np.repeat(weights[j],n_reps))
    allweights = (allweights/sum(allweights))
    
    # weighted average of forecasted yield after being sorted by the metric
    
    #print(allweights)
    #print(zip(out[:,1],allweights))
    a = np.average(out[:,1],weights=allweights)
    
    
    fy_wmean = np.append(fy_wmean, a) # projected weighted mean
    
    # projected weighted standard deviation
    variance = np.average((forecametric-fy_wmean)**2, weights=allweights)  
    fy_wsd = np.sqrt(variance) 
   
    del allweights
    return fy_wmean, fy_wsd
#---------------------------------------------------------------#
def cum_plots(climastartyear, climaendyear, forecastyear, sta_name, wth_path, weights):
    """
    :param climastartyear: the year climatology value start.
    :param climaendyear: the year climatology value end.
    :param forecastyear: the year for which we are going to forecast yield
                         from historical climatic weather.
    :param forecastyear: the year forecast start.
    :param sta_name: the name of the station or point.
    :param wth_path: the path of the wth file (where the weather data is.)

    :return None 
   """

    climayears = np.arange(climastartyear, climaendyear+1)
    #----------------------------------------#
    # warning that certain number of years have been removed from the climatology
    # to make the length divisioble by len(weight) 
    ny_del = len(climayears) % len(weights)
    if ny_del != 0:
         climayears = climayears[:(len(climayears) - ny_del)]
         warnings.warn("The last %s year of climatology years has been removed!" % ny_del)
         print "Only %s ensembles are used!" % len(climayears)
    else:
        climayears = climayears
    #----------------------------------------#    
    path = wth_path
    # read the file containing the forecast year weather data
    forecastyeardata = np.genfromtxt(path+'origi_'+sta_name+'001001'+str(forecastyear)+'.wth',skip_header=4)
    # reading the climatological years data
    for i in range(0,len(climayears)):
        # read the file containing the climatological weather data
        climadata = np.genfromtxt(path+sta_name+'001001'+str(climayears[i])+\
                                  '.wth',skip_header=4)
    
    precip = forecastyeardata[:,4]
    cumprecip = np.cumsum(precip)

    tmin = forecastyeardata[:,3]
    tmax = forecastyeardata[:,2]
    swr = forecastyeardata[:,1]
    
    climarain_all = []
    climatmin_all = []
    climatmax_all = []
    climaswr_all = []
    for i in range(0,len(climayears)):
        # read the file containing the climatological weather data
        climadata = np.genfromtxt(path+sta_name+'001001'+str(climayears[i])+'.wth',skip_header=4)
        climarain = climadata[:,4]
        climarain = np.cumsum(climarain)
        climarain_all = np.append(climarain_all,climarain)

        climatmin = climadata[:,3]
        climatmin_all = np.append(climatmin_all,climatmin)

        climatmax = climadata[:,2]
        climatmax_all = np.append(climatmax_all,climatmax)

        climaswr = climadata[:,1]
        climaswr_all = np.append(climaswr_all,climaswr)
        
    climarain_all = np.reshape(climarain_all,(len(climayears),365))
    climatmin_all = np.reshape(climatmin_all,(len(climayears),365))
    climatmax_all = np.reshape(climatmax_all,(len(climayears),365))
    climaswr_all = np.reshape(climaswr_all,(len(climayears),365))

    av_rain = np.mean(climarain_all,axis=0)
    av_tmin = np.mean(climatmin_all,axis=0)
    av_tmax = np.mean(climatmax_all,axis=0)
    av_swr = np.mean(climaswr_all,axis=0)

    fig = plt.figure()
    sns.set_style("ticks")
    plt.plot(cumprecip,'b',label=str(forecastyear))
    plt.plot(av_rain,'r',label='Climatology average ('+str(climayears[0])+'-'+str(climayears[-1])+')')
    plt.xlabel('DOY',fontsize=14)
    plt.ylabel('Precipitation (mm)',fontsize=14)
    plt.title('Theme: Cumulative rainfall\nLocation: '+sta_name,loc='left', fontsize=14) #%(sta_name)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()                
    plt.savefig('./plot_output/cum_precip.png', dpi=300)
    plt.close()


    fig = plt.figure()
    sns.set_style("ticks")
    plt.plot(tmin,'b',label=str(forecastyear))
    plt.plot(av_tmin,'r',label='Climatology average ('+str(climayears[0])+'-'+str(climayears[-1])+')')
    plt.xlabel('DOY',fontsize=14)
    plt.ylabel('Temperature (C)',fontsize=14)
    plt.title('Theme: Minimum Temperature\nLocation: '+sta_name,loc='left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()                
    plt.savefig('./plot_output/tmin.png', dpi=300)
    plt.close()

    fig = plt.figure()
    sns.set_style("ticks")
    plt.plot(tmax,'b',label=str(forecastyear))
    plt.plot(av_tmax,'r',label='Climatology average ('+str(climayears[0])+'-'+str(climayears[-1])+')')
    plt.xlabel('DOY',fontsize=14)
    plt.ylabel('Temperature (C)',fontsize=14)
    plt.title('Theme: Maximum Temperature\nLocation: '+sta_name,loc='left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()                
    plt.savefig('./plot_output/tmax.png', dpi=300)
    plt.close()

    fig = plt.figure()
    sns.set_style("ticks")
    plt.plot(swr,'b',label=str(forecastyear))
    plt.plot(av_swr,'r',label='Climatology average ('+str(climayears[0])+'-'+str(climayears[-1])+')')
    plt.xlabel('DOY',fontsize=14)
    plt.ylabel('SWR (MJ m-2 day-1)',fontsize=14)
    plt.title('Theme: Short Wave Radiation\nLocation: '+sta_name,loc='left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()                
    plt.savefig('./plot_output/swr.png', dpi=300)
    plt.close()
#---------------------------------------------------------------------------------#
