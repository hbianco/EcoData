# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:14:18 2017

@author: herve.biancotto

@aim: Implements functions to import, standardise and visualise economic data.
"""

#import bloompy as bp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RECESSION_DATA = r'P:\Python Scripts\EcoProject/usrecspan.csv'

def PoP(data, **kwargs):
    """
    Returns the period over period data series.
    :param data: the data to transform.
    :type data: pandas DataFrame of 1 column.
    
    :key word parameters:
    :param per: period corresponding to the period over period transformation. 
    :param freq: frequency of the series to return.
    :type per: string in list ['D', 'M', 'Q', 'A']
    :type freq: string in list ['D', 'M', 'Q', 'A']
    :param method: defines the method to perform the period over period transformation.
    :type method: ['ratio', 'diff']
    """
    #Correspondance tables
    CORRES_B = {'D':1, 'M':21, 'Q':63, 'H':130, 'A':262} #business days
    CORRES_D = {'D':1, 'M':30, 'Q':91, 'H':182, 'A':365} #calendar days
    CORRES_M = {'M':1, 'Q':3, 'H':6, 'A':12}
    CORRES_Q = {'Q':1, 'H':2, 'A':4}
    CORRES_LABEL = {'D':'DoD', 'M':'MoM', 'Q':'QoQ', 'H':'HoH', 'A':'YoY'}
    
    if "per" in kwargs:
        period = kwargs["per"].upper()
    else:
        period = 'A'
        
    if "freq" in kwargs:
        frequency = kwargs["freq"].upper()
    else:
        frequency = 'D'
    
    #map correspondance tables
    if frequency == 'D':
        CORRES = CORRES_D
    elif frequency == 'B':
        CORRES = CORRES_B
    elif frequency == 'M':
        CORRES = CORRES_M
    elif frequency == 'Q':
        CORRES = CORRES_Q
    
    meth = 'ratio'
    if "method" in kwargs:
        if kwargs["method"] == 'diff':
            meth = kwargs["method"]
    
    #Converts series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame(data.name)
    
    # Aligns frequency and shift the second series by the period
    data = data.asfreq(frequency, method='pad')
    data_per = data.shift(CORRES[period])
    if meth == 'ratio':
        result = data/data_per-1
    elif meth == 'diff':
        result = data - data_per
    result.columns = [list(result)[0] + ' ' + CORRES_LABEL[period]]
    # Returns the period over period transformation
    return result


def scale_data(data, mult=1, lead_lag=0, cum_data=False, freq='D'):
    """
    Scales data by applying a multiplier, lagging or leading and cumulating the 
    pandas time series.
    
    :param data: the data to transform.
    :type data: pandas DataFrame.
    
    Named arguments:
    :param mult: multiplier to apply to the series.
    :param mult: a scalar, type float or int.
    
    :param lead_lag: number of period to lead(positive) or lag(negative) data.
    :type lead_lag: signed int.
    """

        #Converts series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame(data.name)
    
    if cum_data:
        value = mult*data.tshift(lead_lag)
        value = value.cumsum()
    else:
        value = mult*data.tshift(lead_lag)
    return value

        

def graph(*args, **kwargs):
    """
    Graphs pandas DataFrame that are passed through.
    Allows overlays of recession periods.
    
    :param data: input data as a pandas DataFrame.
    :type data: pandas DataFrame.
    
    :param recession: Overlays the chart with recession periods.
    :type recession: Boolean.
    
    :param multiple_series: Adds an extra Y axe for each series.
    :type multiple_series: Boolean.
    
    :param chart_size: Size of the ouput chart.
    :type chart_size: tuple of (int, int).
    
    :param title: Gives a title to the chart.
    :type title: str.
    """
    
    
    ## Treat data
    data = [pd.DataFrame(data_set) for data_set in args]
    try:
        assert data is not None
    except AssertionError:
        print("No data to graph")
        return
    
    ## kwargs treatment
    if 'colour' in kwargs:
        if len(kwargs['colour']) != len(args):
            raise Exception('The colour provided should be of same lenght\
            as the data provided')
        try:
            colour = kwargs['colour']
        except Exception as error:
            print(error)
            return
    else:
        colour = {1:'#4a4a4a',
                  2:'#951826', 
                  3:'#ea7600', 
                  4:'#007398', 
                  5:'#33cc78',
                  6:'#f4ee00',
                  7:'#669999', 
                  8:'#ff99ff',
                  9:'#b35900',}
    
    recess = False
    if 'recession' in kwargs:
        if kwargs['recession']:
            rec_data = pd.read_csv(RECESSION_DATA, header=0, names=['date',\
            'recession'], converters={'date':lambda x: pd.to_datetime(x,\
            dayfirst=True)})
            #rec_data = rec_data.asfreq(data[0].index.freqstr)
            recess = True    
    
    extra_ax = False
    sns.set_style('ticks', {'axes.grid': True})
    if 'multiple_series' in kwargs:
        if kwargs['multiple_series']:
            extra_ax = True
            sns.set_style({'axes.grid': False})
            side = 'right' # alternate right and left when adding a new Y axis.
    
    size = None
    if 'chart_size' in kwargs:
            size = kwargs['chart_size']
    
    g_title = ''
    if 'title' in kwargs:
            g_title = kwargs['title']
           
        
    ## Graph creation
    if size is not None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = plt.subplots()
    
    line = plt.plot(data[0], label=data[0].columns[0], color=colour[1]) #base line
    if extra_ax:
        ax.set_ylabel(data[0].columns[0], color=colour[1])
    i = 2 # counter to handle coulours
    for series in data[1:]:
        if extra_ax: #case multiple Y axis
            axn = ax.twinx()   
            axn.set_frame_on(True)#adds to right
            axn.patch.set_visible(False)
            axn.spines[side].set_position(('axes', 1+0.07*(i-2))) #pushes out the new Y axis
            axn.set_ylabel(series.columns[0], color=colour[i])
        line += plt.plot(series, colour[i], label=series.columns[0])
        i+=1 #change colour for next series
    labels = [l.get_label() for l in line] #gets label all together
    plt.legend(line, labels, loc=0)
    
    # Displays recessions as shaded area
    if recess:
        data_rec = rec_data[rec_data['date'] >= data[0].index[0]]
        data_rec.index = range(len(data_rec))
        beg = 0
        if len(data_rec) != 0:
            if len(data_rec)%2 == 1:
                axn = ax.twinx()
                plt.axvspan(data[0].index[0],data_rec[0]\
                        , facecolor='0.5', alpha=0.4)
                axn.grid(False)
                ax.set_yticklabels([])
                sns.despine(ax=axn, left=False, right=False)
                beg = 1
            try:
                for i in range(beg,len(data_rec['date']),2):
                    axn = ax.twinx()
                    plt.axvspan(data_rec.date[i],\
                                data_rec.date[i+1],\
                                facecolor='0.5', alpha=0.4)
                    axn.grid(False)
                    ax.set_yticklabels([]) #no tick label
                    sns.despine(ax=axn, left=False, right=False) # axis not displayed
            except IndexError:
                print('no recession during that period')
    
    #Title
    ax.axes.set_title(g_title)
        
    
if __name__ == "__main__":
    pass
 
#    spx = bp.getHistoricalData('SPX Index','px last','20010101', \
#                                          periodicity='DAILY')
#    conv_like(spx, spx)
#    modSpx = PoP(spx, per='a', freq='m')
#    gdp = bp.getHistoricalData('GDP CHWG Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modGdp = PoP(gdp, per='a', freq='m')
#    pmi = bp.getHistoricalData('NAPMPMI Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modPmi = PoP(pmi, per='a', freq='m')
#    
#    cembi = bp.getHistoricalData('JBCDCOMP Index', 'px_last', '20010101', \
#                                periodicity='DAILY')
#    modCembi = PoP(cembi, per='h', freq='m')
#    modCembi2 = PoP(cembi, per='a', freq='m')
#    
#    usd = bp.getHistoricalData('USTW$ Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modUsd = PoP(usd, per='a', freq='m')
#    
#    graph(modPmi,modCembi, modCembi2,\
#          recession=True, multiple_series=True, title='bla')
    